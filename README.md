# Dream2DGS
Dream2DGS is a tool for text/image to 3D generation based on [**2D Gaussian Splatting**](https://github.com/hbb1/2d-gaussian-splatting) and [**DreamGaussian**](https://dreamgaussian.github.io/).

https://github.com/TingtingLiao/Dream2DGS/assets/45743512/0d7f7d22-1ad3-4a35-b09c-dc5593460db1

## Update log 
<details> <summary>Click to expand</summary>

- **[2024/05/13]**: Fixed segmentation bug.
- **[2024/05/07]**: Released the initial code.

</details>

## Gallery
Here are some rendered gs results from Dream2DGS. From left to right are rendered rgb, rendered normal, normal from depth. 

https://github.com/TingtingLiao/Dream2DGS/assets/45743512/8e16dab5-0369-4405-8d1c-57c205c9c0a7

https://github.com/TingtingLiao/Dream2DGS/assets/45743512/62d9f243-b067-4a08-ac96-44ef65516cfe

https://github.com/TingtingLiao/Dream2DGS/assets/45743512/c14c160c-ace2-4aae-b8ae-54511211220c

https://github.com/TingtingLiao/Dream2DGS/assets/45743512/faf38b0d-0399-44da-ab40-55769476ac13


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

# text-to-3d (sd2.1 + sv3d)  
python main.py --config configs/text_mv.yaml prompt="an ice cream" save_path="an ice cream" 

# extract mesh  
python convert.py --model_path logs/zelda/zelda_gs.ply  

# using normal supervision set dpt=True 
python main.py --config configs/image_sv3d.yaml input=data/zelda_rgba.png save_path=zelda dpt=True 

```

## Acknowledgement 
Special thanks to the projects and their contributors:
* [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)
  
* [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
 
* [SV3D](https://sv3d.github.io/)

## Citation 
If you find Dream2DGS helpful, please consider citing it: 

```bibtex 
@misc{Dream2DGS,
  author = {Tingting Liao},
  title = {Dream2DGS: Text/Image to 3D Generation Tool},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TingtingLiao/dream2dgs}}
}

```
