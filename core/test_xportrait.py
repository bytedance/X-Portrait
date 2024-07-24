# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************
import os
import argparse
import numpy as np
# torch
import torch
from ema_pytorch import EMA
from einops import rearrange
import cv2
# utils
from utils.utils import set_seed, count_param, print_peak_memory
# model
import imageio
from model_lib.ControlNet.cldm.model import create_model
import copy
import glob
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import face_alignment
import sys
from decord import VideoReader
from decord import cpu, gpu

TORCH_VERSION = torch.__version__.split(".")[0]
FP16_DTYPE = torch.float16
print(f"TORCH_VERSION={TORCH_VERSION} FP16_DTYPE={FP16_DTYPE}")

def extract_local_feature_from_single_img(img, fa, remove_local=False, real_tocrop=None, target_res = 512):
    device = img.device
    pred = img.permute([1, 2, 0]).detach().cpu().numpy()

    pred_lmks = img_as_ubyte(resize(pred, (256, 256)))

    try:
        lmks = fa.get_landmarks_from_image(pred_lmks, return_landmark_score=False)[0]
    except:
        print ('undetected faces!!')
        if real_tocrop is None:
            return torch.zeros_like(img) * 2 - 1., [196,196,320,320]
        return torch.zeros_like(img), [196,196,320,320]
    
    halfedge = 32
    left_eye_center = (np.clip(np.round(np.mean(lmks[43:48], axis=0)), halfedge, 255-halfedge) * (target_res / 256)).astype(np.int32)
    right_eye_center = (np.clip(np.round(np.mean(lmks[37:42], axis=0)), halfedge, 255-halfedge) * (target_res / 256)).astype(np.int32)
    mouth_center = (np.clip(np.round(np.mean(lmks[49:68], axis=0)), halfedge, 255-halfedge) * (target_res / 256)).astype(np.int32)

    if real_tocrop is not None:
        pred = real_tocrop.permute([1, 2, 0]).detach().cpu().numpy()

    half_size = target_res // 8 #64
    if remove_local:
        local_viz = pred
        local_viz[left_eye_center[1] - half_size : left_eye_center[1] + half_size, left_eye_center[0] - half_size : left_eye_center[0] + half_size] = 0
        local_viz[right_eye_center[1] - half_size : right_eye_center[1] + half_size, right_eye_center[0] - half_size : right_eye_center[0] + half_size] = 0
        local_viz[mouth_center[1] - half_size : mouth_center[1] + half_size, mouth_center[0] - half_size : mouth_center[0]  + half_size] = 0        
    else:
        local_viz = np.zeros_like(pred)
        local_viz[left_eye_center[1] - half_size : left_eye_center[1] + half_size, left_eye_center[0] - half_size : left_eye_center[0] + half_size] = pred[left_eye_center[1] - half_size : left_eye_center[1] + half_size, left_eye_center[0] - half_size : left_eye_center[0] + half_size]
        local_viz[right_eye_center[1] - half_size : right_eye_center[1] + half_size, right_eye_center[0] - half_size : right_eye_center[0] + half_size] = pred[right_eye_center[1] - half_size : right_eye_center[1] + half_size, right_eye_center[0] - half_size : right_eye_center[0] + half_size]
        local_viz[mouth_center[1] - half_size : mouth_center[1] + half_size, mouth_center[0] - half_size : mouth_center[0]  + half_size] = pred[mouth_center[1] - half_size : mouth_center[1] + half_size, mouth_center[0] - half_size : mouth_center[0] + half_size]

    local_viz = torch.from_numpy(local_viz).to(device)
    local_viz = local_viz.permute([2, 0, 1])
    if real_tocrop is None:
        local_viz = local_viz * 2 - 1.
    return local_viz

def find_best_frame_byheadpose_fa(source_image, driving_video, fa):
    input = img_as_ubyte(resize(source_image, (256, 256)))
    try:
        src_pose_array = fa.get_landmarks_from_image(input, return_landmark_score=False)[0]
    except:
        print ('undetected faces in the source image!!')
        src_pose_array = np.zeros((68,2))
    if len(src_pose_array) == 0:
        return 0
    min_diff = 1e8
    best_frame = 0

    for i in range(len(driving_video)):
        frame = img_as_ubyte(resize(driving_video[i], (256, 256)))
        try:
            drv_pose_array = fa.get_landmarks_from_image(frame, return_landmark_score=False)[0]
        except:
            print ('undetected faces in the %d-th driving image!!'%i)
            drv_pose_array = np.zeros((68,2))
        diff = np.sum(np.abs(np.array(src_pose_array)-np.array(drv_pose_array)))
        if diff < min_diff:
            best_frame = i
            min_diff = diff   
    
    return best_frame

def adjust_driving_video_to_src_image(source_image, driving_video, fa, nm_res, nmd_res, best_frame=-1):
    if best_frame == -2:
        return [resize(frame, (nm_res, nm_res)) for frame in driving_video], [resize(frame, (nmd_res, nmd_res)) for frame in driving_video]
    src = img_as_ubyte(resize(source_image[..., :3], (256, 256)))
    if  best_frame >= len(source_image):
        raise ValueError(
            f"please specify one frame in driving video of which the pose match best with the pose of source image"
        )

    if best_frame < 0:
        best_frame = find_best_frame_byheadpose_fa(src, driving_video, fa)

    print ('Best Frame: %d' % best_frame)
    driving = img_as_ubyte(resize(driving_video[best_frame], (256, 256)))

    src_lmks = fa.get_landmarks_from_image(src, return_landmark_score=False)
    drv_lmks = fa.get_landmarks_from_image(driving, return_landmark_score=False)

    if (src_lmks is None) or (drv_lmks is None):
        return [resize(frame, (nm_res, nm_res)) for frame in driving_video], [resize(frame, (nmd_res, nmd_res)) for frame in driving_video]
    src_lmks = src_lmks[0]
    drv_lmks = drv_lmks[0]
    src_centers = np.mean(src_lmks, axis=0)
    drv_centers = np.mean(drv_lmks, axis=0)
    edge_src = (np.max(src_lmks, axis=0) - np.min(src_lmks, axis=0))*0.5
    edge_drv = (np.max(drv_lmks, axis=0) - np.min(drv_lmks, axis=0))*0.5

    #matching three points 
    src_point=np.array([[src_centers[0]-edge_src[0],src_centers[1]-edge_src[1]],[src_centers[0]+edge_src[0],src_centers[1]-edge_src[1]],[src_centers[0]-edge_src[0],src_centers[1]+edge_src[1]],[src_centers[0]+edge_src[0],src_centers[1]+edge_src[1]]]).astype(np.float32)
    dst_point=np.array([[drv_centers[0]-edge_drv[0],drv_centers[1]-edge_drv[1]],[drv_centers[0]+edge_drv[0],drv_centers[1]-edge_drv[1]],[drv_centers[0]-edge_drv[0],drv_centers[1]+edge_drv[1]],[drv_centers[0]+edge_drv[0],drv_centers[1]+edge_drv[1]]]).astype(np.float32)
   
    adjusted_driving_video = []
    adjusted_driving_video_hd = []
    
    for frame in driving_video:
        frame_ld = resize(frame, (nm_res, nm_res))
        frame_hd = resize(frame, (nmd_res, nmd_res))
        zoomed=cv2.warpAffine(frame_ld, cv2.getAffineTransform(dst_point[:3], src_point[:3]), (nm_res, nm_res))
        zoomed_hd=cv2.warpAffine(frame_hd, cv2.getAffineTransform(dst_point[:3] * 2, src_point[:3] * 2), (nmd_res, nmd_res))
        adjusted_driving_video.append(zoomed)
        adjusted_driving_video_hd.append(zoomed_hd)
    
    return adjusted_driving_video, adjusted_driving_video_hd

def x_portrait_data_prep(source_image_path, driving_video_path, device, best_frame_id=0, start_idx = 0, num_frames=0, skip=1, output_local=False, more_source_image_pattern="", target_resolution = 512):
    source_image = imageio.imread(source_image_path)
    if '.mp4' in driving_video_path:
        reader = imageio.get_reader(driving_video_path)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
    else:
        driving_video = [imageio.imread(driving_video_path)[...,:3]]
        fps = 1

    nmd_res = target_resolution
    nm_res = 256
    source_image_hd = resize(source_image, (nmd_res, nmd_res))[..., :3]

    if more_source_image_pattern:
        more_source_paths = glob.glob(more_source_image_pattern)
        more_sources_hd = []
        for more_source_path in more_source_paths:
            more_source_image = imageio.imread(more_source_path)
            more_source_image_hd = resize(more_source_image, (nmd_res, nmd_res))[..., :3]
            more_source_hd = torch.tensor(more_source_image_hd[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            more_source_hd = more_source_hd.to(device)
            more_sources_hd.append(more_source_hd)
        more_sources_hd = torch.stack(more_sources_hd, dim = 1) 
    else:
        more_sources_hd = None

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True, device='cuda')

    driving_video, driving_video_hd = adjust_driving_video_to_src_image(source_image, driving_video, fa, nm_res, nmd_res, best_frame_id)

    if num_frames == 0:
        end_idx = len(driving_video)
    else:
        num_frames = min(len(driving_video), num_frames)
        end_idx = start_idx + num_frames * skip
    
    driving_video = driving_video[start_idx:end_idx][::skip]
    driving_video_hd = driving_video_hd[start_idx:end_idx][::skip]
    num_frames = len(driving_video)

    with torch.no_grad():
        real_source_hd = torch.tensor(source_image_hd[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        real_source_hd = real_source_hd.to(device)

        driving_hd = torch.tensor(np.array(driving_video_hd).astype(np.float32)).permute(0, 3, 1, 2).to(device)

        local_features = []
        raw_drivings=[]

        for frame_idx in range(0, num_frames):
            raw_drivings.append(driving_hd[frame_idx:frame_idx+1] * 2 - 1.)
            if output_local:
                local_feature_img = extract_local_feature_from_single_img(driving_hd[frame_idx], fa,target_res=nmd_res)
                local_features.append(local_feature_img)


    batch_data = {}
    batch_data['fps'] = fps
    real_source_hd = real_source_hd * 2 - 1
    batch_data['sources'] = real_source_hd[:, None, :, :, :].repeat([num_frames, 1, 1, 1, 1]) 
    if more_sources_hd is not None:
        more_sources_hd = more_sources_hd * 2 - 1
        batch_data['more_sources'] = more_sources_hd.repeat([num_frames, 1, 1, 1, 1])

    raw_drivings = torch.stack(raw_drivings, dim = 0)
    batch_data['conditions'] = raw_drivings
    if output_local:
        batch_data['local'] = torch.stack(local_features, dim = 0)

    return batch_data

# You can now use the modified state_dict without the deleted keys
def load_state_dict(model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = torch.load(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    model.load_state_dict(state_dict, strict=strict)
    del state_dict   

def get_cond_control(args, batch_data, control_type, device, start, end, model=None, batch_size=None, train=True, key=0):

    control_type = copy.deepcopy(control_type)
    vae_bs = 16
    if control_type == "appearance_pose_local_mm":
        src = batch_data['sources'][start:end, key].cuda()
        c_cat_list = batch_data['conditions'][start:end].cuda()
        cond_image = []
        for k in range(0, end-start, vae_bs):
            cond_image.append(model.get_first_stage_encoding(model.encode_first_stage(src[k:k+vae_bs])))
        cond_image = torch.concat(cond_image, dim=0)
        cond_img_cat = cond_image
        p_local = batch_data['local'][start:end].cuda()    
        print ('Total frames:{}'.format(cond_img_cat.shape))
        more_cond_imgs = []
        if 'more_sources' in batch_data:
            num_additional_cond_imgs = batch_data['more_sources'].shape[1]
            for i in range(num_additional_cond_imgs):
                m_cond_img = batch_data['more_sources'][start:end, i]
                m_cond_img = model.get_first_stage_encoding(model.encode_first_stage(m_cond_img))
                more_cond_imgs.append([m_cond_img.to(device)])

        return [cond_img_cat.to(device), c_cat_list, p_local, more_cond_imgs]    
    else:
        raise NotImplementedError(f"cond_type={control_type} not supported!")

def visualize_mm(args, name, batch_data, infer_model, nSample, local_image_dir, num_mix=4, preset_output_name=''):
    driving_video_name = os.path.basename(batch_data['video_name']).split('.')[0]
    source_name = os.path.basename(batch_data['source_name']).split('.')[0]

    if not os.path.exists(local_image_dir):
        os.mkdir(local_image_dir)

    uc_scale = args.uc_scale
    if preset_output_name:
        preset_output_name = preset_output_name.split('.')[0]+'.mp4'
        output_path = f"{local_image_dir}/{preset_output_name}"
    else:
        output_path = f"{local_image_dir}/{name}_{args.control_type}_uc{uc_scale}_{source_name}_by_{driving_video_name}_mix{num_mix}.mp4"

    infer_model.eval()

    gene_img_list = []
    
    _, _, ch, h, w = batch_data['sources'].shape

    vae_bs = 16

    if args.initial_facevid2vid_results:
        facevid2vid = []
        facevid2vid_results = VideoReader(args.initial_facevid2vid_results, ctx=cpu(0))
        for frame_id in range(len(facevid2vid_results)):
            frame = cv2.resize(facevid2vid_results[frame_id].asnumpy(),(512,512)) / 255
            facevid2vid.append(torch.from_numpy(frame * 2 - 1).permute(2,0,1))
        cond = torch.stack(facevid2vid)[:nSample].float().to(args.device)
        pre_noise=[]
        for i in range(0, nSample, vae_bs):
            pre_noise.append(infer_model.get_first_stage_encoding(infer_model.encode_first_stage(cond[i:i+vae_bs])))
        pre_noise = torch.cat(pre_noise, dim=0)
        pre_noise = infer_model.q_sample(x_start = pre_noise, t = torch.tensor([999]).to(pre_noise.device))
    else:
        cond = batch_data['sources'][:nSample].reshape([-1, ch, h, w])
        pre_noise=[]
        for i in range(0, nSample, vae_bs):
            pre_noise.append(infer_model.get_first_stage_encoding(infer_model.encode_first_stage(cond[i:i+vae_bs])))
        pre_noise = torch.cat(pre_noise, dim=0)
        pre_noise = infer_model.q_sample(x_start = pre_noise, t = torch.tensor([999]).to(pre_noise.device))

    text = ["" for _ in range(nSample)]
    
    all_c_cat = get_cond_control(args, batch_data, args.control_type, args.device, start=0, end=nSample, model=infer_model, train=False)
    cond_img_cat = [all_c_cat[0]]
    pose_cond_list = [rearrange(all_c_cat[1], "b f c h w -> (b f) c h w")]
    local_pose_cond_list = [all_c_cat[2]]

    c_cross = infer_model.get_learned_conditioning(text)[:nSample]
    uc_cross = infer_model.get_unconditional_conditioning(nSample)

    c = {"c_crossattn": [c_cross], "image_control": cond_img_cat}
    if "appearance_pose" in args.control_type:
        c['c_concat'] = pose_cond_list
    if "appearance_pose_local" in args.control_type:
        c["local_c_concat"] = local_pose_cond_list
    
    if len(all_c_cat) > 3 and len(all_c_cat[3]) > 0:
        c['more_image_control'] = all_c_cat[3]

    if args.control_mode == "controlnet_important":
        uc = {"c_crossattn": [uc_cross]}
    else:
        uc = {"c_crossattn": [uc_cross], "image_control":cond_img_cat}

    if "appearance_pose" in args.control_type:
        uc['c_concat'] = [torch.zeros_like(pose_cond_list[0])]

    if "appearance_pose_local" in args.control_type:
        uc["local_c_concat"] = [torch.zeros_like(local_pose_cond_list[0])]

    if len(all_c_cat) > 3 and len(all_c_cat[3]) > 0:
        uc['more_image_control'] = all_c_cat[3]

    if args.wonoise:
        c['wonoise'] = True
        uc['wonoise'] = True
    else:
        c['wonoise'] = False
        uc['wonoise'] = False
        
    noise = pre_noise.to(c_cross.device)

    with torch.cuda.amp.autocast(enabled=args.use_fp16, dtype=FP16_DTYPE):
        infer_model.to(args.device)
        infer_model.eval()

        gene_img, _ = infer_model.sample_log(cond=c,
                                    batch_size=args.num_drivings, ddim=True,
                                    ddim_steps=args.ddim_steps, eta=args.eta,
                                    unconditional_guidance_scale=uc_scale,
                                    unconditional_conditioning=uc,
                                    inpaint=None,
                                    x_T=noise,
                                    num_overlap=num_mix,
                                    )

        for i in range(0, nSample, vae_bs):
            gene_img_part = infer_model.decode_first_stage( gene_img[i:i+vae_bs] )
            gene_img_list.append(gene_img_part.float().clamp(-1, 1).cpu())

    _, c, h, w = gene_img_list[0].shape  

    cond_image = batch_data["conditions"].reshape([-1,c,h,w])[:nSample].cpu()
    l_cond_image = batch_data["local"].reshape([-1,c,h,w])[:nSample].cpu()
    orig_image = batch_data["sources"][:nSample, 0].cpu()

    output_img = torch.cat(gene_img_list + [cond_image.cpu()]+[l_cond_image.cpu()]+[orig_image.cpu()]).float().clamp(-1,1).add(1).mul(0.5)

    num_cols = 4
    output_img = output_img.reshape([num_cols, 1, nSample, c, h, w]).permute([1, 0, 2, 3, 4,5])

    output_img = output_img.permute([2, 3, 0, 4, 1, 5]).reshape([-1, c,  h,  num_cols * w])
    output_img = torch.permute(output_img, [0, 2, 3, 1])
    
    output_img = output_img.data.cpu().numpy()
    output_img = img_as_ubyte(output_img)
    imageio.mimsave(output_path, output_img[:,:,:512], fps=batch_data['fps'], quality=10, pixelformat='yuv420p', codec='libx264')

def main(args):
    
    # ******************************
    # initialize training
    # ******************************
    args.world_size = 1
    args.local_rank = 0
    args.rank = 0
    args.device = torch.device("cuda", args.local_rank)

    # set seed for reproducibility
    set_seed(args.seed)

    # ******************************
    # create model
    # ******************************
    model = create_model(args.model_config).cpu()
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    model.to(args.local_rank)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.local_rank == 0:
        print('Total base  parameters {:.02f}M'.format(count_param([model])))
    if args.ema_rate is not None and args.ema_rate > 0 and args.rank == 0:
        print(f"Creating EMA model at ema_rate={args.ema_rate}")
        model_ema = EMA(model, beta=args.ema_rate, update_after_step=0, update_every=1)
    else:
        model_ema = None

    # ******************************
    # load pre-trained models
    # ******************************
    if args.resume_dir is not None:
        if args.local_rank == 0:
            load_state_dict(model, args.resume_dir, strict=False)
    else:
        print('please privide the correct resume_dir!')
        exit()
    
    # ******************************
    # create DDP model
    # ******************************
    if args.compile and TORCH_VERSION == "2":
        model = torch.compile(model)
    
    torch.cuda.set_device(args.local_rank)
    print_peak_memory("Max memory allocated after creating DDP", args.local_rank)
    infer_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        driving_videos = glob.glob(args.driving_video)
        for driving_video in driving_videos:
            print ('working on {}'.format(os.path.basename(driving_video)))
            infer_batch_data = x_portrait_data_prep(args.source_image, driving_video, args.device, args.best_frame, start_idx = args.start_idx, num_frames = args.out_frames, skip=args.skip, output_local=True)
            infer_batch_data['video_name'] = os.path.basename(driving_video)
            infer_batch_data['source_name'] = args.source_image
            nSample = infer_batch_data['sources'].shape[0]
            visualize_mm(args, "inference", infer_batch_data, infer_model, nSample=nSample, local_image_dir=args.output_dir, num_mix=args.num_mix)


if __name__ == "__main__":

    str2bool = lambda arg: bool(int(arg))
    parser = argparse.ArgumentParser(description='Control Net training')
    ## Model
    parser.add_argument('--model_config', type=str, default="model_lib/ControlNet/models/cldm_v15_video_appearance.yaml",
                        help="The path of model config file")
    parser.add_argument('--reinit_hint_block', action='store_true', default=False,
                        help="Re-initialize hint blocks for channel mis-match")
    parser.add_argument('--sd_locked', type =str2bool, default=True,
                        help='Freeze parameters in original stable-diffusion decoder')
    parser.add_argument('--only_mid_control', type =str2bool, default=False,
                        help='Only control middle blocks')
    parser.add_argument('--control_type', type=str, default="appearance_pose_local_mm",
                        help='The type of conditioning')
    parser.add_argument("--control_mode", type=str, default="controlnet_important",
                        help="Set controlnet is more important or balance.")
    parser.add_argument('--wonoise', action='store_false', default=True,
                        help='Use with referenceonly, remove adding noise on reference image')
 
    ## Training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed for initialization')
    parser.add_argument('--use_fp16', action='store_false', default=True,
                        help='Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit')
    parser.add_argument('--compile', type=str2bool, default=False,
                        help='compile model (for torch 2)')
    parser.add_argument('--eta', type = float, default = 0.0,
                        help='eta during DDIM Sampling')
    parser.add_argument('--ema_rate', type = float, default = 0,
                        help='rate for ema')
    ## inference
    parser.add_argument("--initial_facevid2vid_results", type=str, default=None,
                    help="facevid2vid results for noise initialization")
    parser.add_argument('--ddim_steps', type = int, default = 1,
                        help='denoising steps')
    parser.add_argument('--uc_scale', type = int, default = 5,
                        help='cfg')
    parser.add_argument("--num_drivings", type = int, default = 16,
                        help="Number of driving images in a single sequence of video.")
    parser.add_argument("--output_dir", type=str, default=None, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="The resume directory where the model checkpoints will be loaded.")
    parser.add_argument("--source_image", type=str, default="",
                        help="The source image for neural motion.")                  
    parser.add_argument("--more_source_image_pattern", type=str, default="",
                        help="The source image for neural motion.")   
    parser.add_argument("--driving_video", type=str, default="",
                        help="The source image mask for neural motion.")                 
    parser.add_argument('--best_frame', type=int, default=0,
                        help='best matching frame index')     
    parser.add_argument('--start_idx', type=int, default=0,
                        help='starting frame index')   
    parser.add_argument('--skip', type=int, default=1,
                        help='skip frame')  
    parser.add_argument('--num_mix', type=int, default=4,
                        help='num overlapping frames')  
    parser.add_argument('--out_frames', type=int, default=0,
                        help='num frames')  
    args = parser.parse_args()

    main(args)
    