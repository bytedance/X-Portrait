<!-- # magic-edit.github.io -->

<p align="center">

  <h2 align="center">X-Portrait: Expressive Portrait Animation with Hierarchical Motion Attention</h2>
  <p align="center">
                <a href="https://scholar.google.com/citations?user=FV0eXhQAAAAJ&hl=en">You Xie</a>,
                <a href="https://hongyixu37.github.io/homepage/">Hongyi Xu</a>,
                <a href="https://guoxiansong.github.io/homepage/index.html">Guoxian Song</a>,
                <a href="https://chaowang.info/">Chao Wang</a>,
                <a href="https://seasonsh.github.io/">Yichun Shi</a>,
                <a href="http://linjieluo.com/">Linjie Luo</a>
    <br>
    <b>&nbsp;  ByteDance Inc. </b>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2403.15931"><img src='https://img.shields.io/badge/arXiv-X--Portrait-red' alt='Paper PDF'></a>
        <a href='https://byteaigc.github.io/x-portrait/'><img src='https://img.shields.io/badge/Project_Page-X--Portrait-green' alt='Project Page'></a>
        <a href='https://youtu.be/VGxt5XghRdw'>
        <img src='https://img.shields.io/badge/YouTube-X--Portrait-rgb(255, 0, 0)' alt='Youtube'></a>
    <br>
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="assets/teaser/teaser.png">
    </td>
    </tr>
  </table>

This repository contains the video generation code of SIGGRAPH 2024 paper [X-Portrait](https://arxiv.org/pdf/2403.15931). 

## Installation
Note: Python 3.9 and Cuda 11.8 are required.
```shell
bash env_install.sh
```

## Model
Please download pre-trained model from [here](https://drive.google.com/drive/folders/1Bq0n-w1VT5l99CoaVg02hFpqE5eGLo9O?usp=sharing), and save it under "checkpoint/"

## Testing
```shell
bash scripts/test_xportrait.sh
```
parameters:  
**model_config**: config file of the corresponding model  
**output_dir**: output path for generated video  
**source_image**: path of source image  
**driving_video**: path of driving video  
**best_frame**: specify the frame index in the driving video where the head pose best matches the source image (precision of best_frame index might affect the final quality)  
**out_frames**: number of generation frames  
**skip**: selecting one frame while skipping a number of frames in the driving video during inference  
**num_mix**: number of overlapping frames when applying prompt travelling during inference  
**ddim_steps**: number of inference steps (e.g., 30 steps for ddim)     

## Performance Boost
**efficiency**: Our model is compatible with LCM LoRA (https://huggingface.co/latent-consistency/lcm-lora-sdv1-5), which helps reduce the number of inference steps.  
**expressiveness**: Expressiveness of the results could be boosted if results of other face reenactment approaches, e.g., face vid2vid, could be provided to generate inital noise via parameter "--initial_facevid2vid_results".  

## 🎓 Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@inproceedings{xie2024x,
  title={X-Portrait: Expressive Portrait Animation with Hierarchical Motion Attention},
  author={Xie, You and Xu, Hongyi and Song, Guoxian and Wang, Chao and Shi, Yichun and Luo, Linjie},
  journal={arXiv preprint arXiv:2403.15931},
  year={2024}
}
```
