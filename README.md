# Dream2DGS
An implementation of the text-to-3D model based on [**2D Gaussian Splatting**](https://github.com/hbb1/2d-gaussian-splatting) and [**DreamGaussian**](https://dreamgaussian.github.io/).
 
 
https://github.com/TingtingLiao/Dream2DGS/assets/45743512/8e16dab5-0369-4405-8d1c-57c205c9c0a7


## Install

```bash
conda create -n dream2dgs python=3.9 

# torch2.3.0+cu12.1 
pip3 install torch torchvision torchaudio

# xformers  
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
 
pip install -r requirements.txt
  
# mvdream 
pip install git+https://github.com/bytedance/MVDream

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/
 
pip install -e submodules/diff-surfel-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/generative_models

# download pretriained models  
mkdir checkpoints && cd checkpoints 
wget https://huggingface.co/stabilityai/sv3d/blob/main/sv3d_p.safetensors  # sv3d_p.safetensors
gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
cd ..
```

## Usage 
```bash   
# image-to-3d (sv3d)  
python main.py --config configs/image_sv3d.yaml input=data/zelda_rgba.png save_path=zelda
   
# extract mesh  
python convert.py --model_path logs/zelda/zelda.ply  
```

# Acknowledgement 

* [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)
    ```
    @inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
    }
    ```

* [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
   ```
   @article{tang2023dreamgaussian,
    title={DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation},
    author={Tang, Jiaxiang and Ren, Jiawei and Zhou, Hang and Liu, Ziwei and Zeng, Gang},
    journal={arXiv preprint arXiv:2309.16653},
    year={2023}
    } 
    ```
