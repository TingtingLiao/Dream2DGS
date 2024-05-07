# Dream2DGS
An implementation of the text-to-3D model based on [**2D Gaussian Splatting**](https://github.com/hbb1/2d-gaussian-splatting) and [**DreamGaussian**](https://dreamgaussian.github.io/).



https://github.com/TingtingLiao/Dream2DGS/assets/45743512/44bff53e-e4c4-45da-99ca-588dcded798f


## Install

```bash
conda create -n dream2dgs python=3.9 

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
 
pip install -r requirements.txt

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

cd submodules/
pip install -e diff-surfel-rasterization
pip install -e simple-knn

# To use MVdream, also install:
pip install git+https://github.com/bytedance/MVDream

# To use ImageDream, also install:
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream
```

## Usage 
```bash   
# image-to-3d (zero123)  
python main.py --config configs/image.yaml input=data/csm_luigi_rgba.png save_path=csm_luigi 

# extract mesh  
python convert.py --model_path logs/csm_luigi/csm_luigi.ply  

# text-to-3d (mvdream)
python main.py --config configs/text_mv.yaml prompt="a plush toy of a corgi nurse" save_path=corgi_nurse  
 
 
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
