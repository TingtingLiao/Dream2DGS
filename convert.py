import torch
import argparse  
import os
import numpy as np
from tqdm import tqdm
from fpsample import fps_sampling
import fast_simplification
from os import makedirs
from gs_renderer import Renderer, MiniCam
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
# from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
from gs_model import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh, Mesh
from utils.render_utils import generate_path, create_videos
from utils.mesh_renderer import MeshRenderer
from utils.cam_utils import orbit_camera, OrbitCamera, look_at 
import open3d as o3d


def fibonacci_sphere(samples=1, randomize=True):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2. / samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        points.append([x, y, z])
    return np.array(points)


def orbit_camera_fibonacci(num_samples, radius=2.5, is_degree=True, target=None, opengl=True, render_resolution=512, fov=49.1):
    cam_positions = fibonacci_sphere(num_samples, randomize=False)   

    cameras = []
    for campos in cam_positions: 
        elevation = np.arcsin(campos[1] / radius)  # y
        azimuth = np.arctan2(campos[2], campos[0])  # z, x

        if is_degree:
            elevation = np.rad2deg(elevation)
            azimuth = np.rad2deg(azimuth)
        
        camera_matrix = orbit_camera(elevation, azimuth, radius, is_degree, target, opengl)
 
        cur_cam = MiniCam(
            camera_matrix, render_resolution, render_resolution,  fov,  fov,  0.1, 100
        )
        cameras.append(cur_cam)
    return cameras

def generate_cameras(render_resolution=512, fov=49.1, radius=2.5, num_cameras=100, pitch = -20): 
    yaws = torch.linspace(0, 360, num_cameras) 
    pitchs = torch.linspace(0, 360, num_cameras) 

    cameras = []
    fov = np.deg2rad(fov)
    for yaw, _ in zip(yaws, pitchs):  
        pose = orbit_camera(pitch, yaw, radius)
        cur_cam = MiniCam(
            pose, 
            render_resolution, 
            render_resolution, 
            fov, 
            fov, 
            0.1, 100
        )
        
        cameras.append(cur_cam)
    return cameras


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--iteration", default=100, type=int)  
    parser.add_argument("--voxel_size", default=0.008, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=4.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--num_cluster", default=1000, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--optimize_texture", action="store_true", help='Mesh: optimize texture for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    
    device = torch.device("cuda")
    args, extras = parser.parse_known_args() 
    iteration = args.iteration
    save_dir = os.path.dirname(args.model_path)
    os.makedirs(save_dir, exist_ok=True)

    # Load the model
    gs_renderer = Renderer(sh_degree=3, white_background=True)  
    gs_renderer.gaussians.load_ply(args.model_path) 
    gaussExtractor = GaussianExtractor(gs_renderer)    
      
    n_fames = 200 
    radius = 2 
    fov = 49.1 
    cameras = orbit_camera_fibonacci(n_fames, render_resolution=512, fov=49.1)  
    # cameras = generate_cameras(render_resolution=512, fov=49.1, radius=radius, num_cameras=n_fames, pitch = -20)
    # render frames 
    gaussExtractor.reconstruction(cameras)
    # extract mesh 
    if args.unbounded: 
        mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
    else:
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=args.voxel_size, sdf_trunc=0.05, depth_trunc=args.depth_trunc)
    mesh = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    # save mesh 
    save_path = os.path.join(save_dir, 'fused.ply') 
    o3d.io.write_triangle_mesh(save_path, mesh) 
    print("mesh post processed saved at {}".format(save_path))

    # optimize uv texture  
    if args.optimize_texture:
        mesh_renderer = MeshRenderer(
            near=0.01,
            far=100,
            ssaa=1,
            texture_filter='linear-mipmap-linear'
            ).to(device)

        print('Start optimizing texture...')

        # simplify mesh
        mesh = Mesh.load(save_path, auto_uv=False, device='cpu')
        new_v, new_f = fast_simplification.simplify(mesh.v.numpy(), mesh.f.numpy(), target_reduction=0.3)
        mesh = Mesh(v=torch.tensor(new_v, dtype=torch.float32), f=torch.tensor(new_f))
        print("new mesh ", new_v.shape, new_f.shape)
        mesh.auto_normal()
        mesh.auto_uv()
        mesh = mesh.to(device)
 
        num_cameras = 32 
        cameras = generate_cameras(render_resolution=512, fov=fov, radius=radius, num_cameras=num_cameras, pitch = 0)
        gaussExtractor.reconstruction(cameras)
        bake_alphas = gaussExtractor.alphamaps.permute(0, 2, 3, 1).float().to(device) # [num_cameras, H, W, 1]
        bake_images = gaussExtractor.rgbmaps.permute(0, 2, 3, 1).float().to(device) # [num_cameras, H, W, 3]
        bake_images = (bake_images  - (1 - bake_alphas)) / bake_alphas.clamp(min=1e-6)
          
        c2ws = torch.stack([cam.world_view_transform.transpose(0, 1).inverse() for cam in cameras], dim=0).float().to(device=device)
        f = 0.5 / np.tan(np.deg2rad(fov / 2)) 
        intrinsics = torch.tensor([f, f, 0.5, 0.5], device=device).float() 

        save_path = f"{save_dir}/model.obj"
        texture = mesh_renderer.bake_multiview(mesh, bake_images, bake_alphas, c2ws, intrinsics)
        mesh.albedo = texture
 
        mesh.textureless = False
        mesh.write(save_path, flip_yz=False)
        print(f"Save optimized mesh at {save_path}")