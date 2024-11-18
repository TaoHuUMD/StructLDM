
<div align="center">

# <b>StructLDM</b>: Structured Latent Diffusion for 3D Human Generation 

<h2>ECCV 2024</h2>

[Tao Hu](https://taohuumd.github.io/), [Fangzhou Hong](https://github.com/hongfz16), [Ziwei Liu](https://liuziwei7.github.io/)

[S-Lab, Nanyang Technological University](https://www.ntu.edu.sg/s-lab)

### [Project Page](https://taohuumd.github.io/projects/StructLDM/) · [Paper](https://arxiv.org/pdf/2404.01241) · [Video](https://www.youtube.com/watch?v=9GKdWVXcNqA)

</div>

## Introduction
We propose StructLDM, a structured latent diffusion model that learns 3D human generations from 2D images.

<img src='docs/figs/teaser.jpg'>

StructLDM generates diverse view-consistent humans, and supports different levels of controllable generation and editing, such as compositional generations by blending the five selected parts from a), and part-aware editing such as identity swapping, local clothing editing, 3D virtual try-on, etc. Note that the generations and editing are clothing-agnostic without clothing types or masks conditioning.

<img src='docs/figs/ezgif-4-0a5af9cccc.gif'>
Generations on RenderPeople.

## Installation
NVIDIA GPUs are required for this project. We have trained and tested code on NVIDIA V100.  We recommend using anaconda to manage the python environments.

```bash
conda create --name structldm python=3.9
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install -r requirements.txt
```

## Implementation 

### Download Models & Assets & Datasets

Download sample data, necessary assets, and pretrained models from [OneDrive](https://1drv.ms/f/c/cd958c29ffd57ddb/EhBoSdYizdVPkjyy85LDkM8BQzbRM1BFIlkrxtwwbH1_hA?e=LvZmCU). Put them in *DATA_DIR/result/trained_model* and *DATA_DIR/asset* respectively. *DATA_DIR* is specified as *./data* in default.
 
Register and download SMPL models [here](https://smpl.is.tue.mpg.de/). Put them in the folder *smpl_data*.

The folder structure should look like

```
DATA_DIR
├── dataset
    ├──renderpeople/
└── asset/
    ├── smpl_data/
        └── SMPL_NEUTRAL.pkl
    ├── uv_sampler/
    ├── uv_table.npy
    ├── smpl_uv.obj
    ├── smpl_template_sdf.npy
    ├── sample_data.pkl
├── result/
    ├── trained_model/modelname/
        └──decoder_xx, diffusion_xx
        ├──samples/
    ├── test_output

```

### Commands

Generating 3D humans via (e.g., models trained on RenderPeople).
```bash
bash scripts/renderpeople.sh gpu_ids
```
The generation results will be found in *DATA_DIR/result/test_output*.

The training script of latent diffusion can be found in *struct_diffusion*.
```bash
bash struct_diffusion/scripts/exec.sh "train" gpu_ids
```
Trained models will be stored in *DATA_DIR/result/trained_model/modelname/diffusion_xx.pt*.

Refer to the downloaded sample data at ./data/dataset/renderpeople to prepare your own dataset, and modify the corresponding path in the config file.

The inference script of latent diffusion can be found in *struct_diffusion*.
```bash
bash struct_diffusion/scripts/test.sh gpu_ids
```
Samples will be stored in *DATA_DIR/result/trained_model/modelname/samples*. 

## License
Distributed under the S-Lab License. See `LICENSE` for more information.

SMPL-X related files are subject to the license of [SMPL-X](https://smpl-x.is.tue.mpg.de/modellicense.html).

## Citation
If you find our code or paper is useful to your research, please consider citing:
```bibtex
@misc{hu2024structldm,
      title={StructLDM: Structured Latent Diffusion for 3D Human Generation}, 
      author={Tao Hu and Fangzhou Hong and Ziwei Liu},
      year={2024},
      eprint={2404.01241},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
  }
```

## Acknowledgements
The structured diffusion model is implemented on top of the [Latent-Diffusion](https://github.com/CompVis/latent-diffusion).
