#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh
import torch.nn.functional as F
import xatlas

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
   
    # n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    # n_cluster = max(n_cluster, 500) # filter meshes smaller than 50
    
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 500
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()

    # mesh_0 = mesh_0.filter_smooth_simple(number_of_iterations=10)
    # mesh_0 = mesh_0.compute_vertex_normals()

    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width, 
                    height=viewpoint_cam.image_height, 
                    cx = viewpoint_cam.image_width/2,
                    cy = viewpoint_cam.image_height/2,
                    fx = viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx / 2.)),
                    fy = viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy / 2.))
                    )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    
    def __init__(self, gs_renderer):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        
        self.render = gs_renderer 
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.points = []
        self.viewpoint_stack = []

 

    @torch.no_grad()
    def reconstruction(self, cameras):
        """
        reconstruct radiance field given cameras
        """
        self.clean() 
        self.viewpoint_stack = cameras
        for viewpoint_cam in tqdm(cameras): 
           
            render_pkg = self.render.render(viewpoint_cam)
 
            rgb = render_pkg['image']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            point = render_pkg['surf_point']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
            self.points.append(point.cpu())
        
        self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        self.depthmaps = torch.stack(self.depthmaps, dim=0)
        self.alphamaps = torch.stack(self.alphamaps, dim=0)
        self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.points = torch.stack(self.points, dim=0)

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=False):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        # depth_trunc = 4 
        # voxel_size = 4.0 / 512
        # sdf_trunc = 0.04 

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, 
                convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        #TODO: support color mesh exporting

        sdf_trunc: truncation value
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, normalmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sampled_normal = torch.nn.functional.grid_sample(normalmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, sampled_normal, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                samples = inv_contraction(samples)
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, normal, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    normalmap = self.depth_normals[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        center = torch.from_numpy(center).float().cuda()
        normalize = lambda x: (x - center) / radius
        unnormalize = lambda x: (x * radius) + center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True) 
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"): 
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))


class Mesh:
    def __init__(
            self,
            v=None,
            f=None,
            vn=None,
            fn=None,
            vt=None,
            ft=None,
            vc=None,
            albedo=None,
            ks=None,
            bump=None,
            device=None,
            textureless=False
    ):
        self.device = device
        self.v = v
        self.vn = vn
        self.vt = vt
        self.vc = vc
        self.f = f
        self.fn = fn
        self.ft = ft
        self.face_normals = None
        # only support a single albedo
        white_image = torch.ones((1024, 1024, 3), dtype=torch.float32)
        self.albedo = albedo
        self.ks = ks
        self.bump = bump
        self.textureless = textureless

        self.ori_center = 0
        self.ori_scale = 1

    def detach(self):
        attrs = ['v', 'vn', 'vt', 'vc', 'f', 'fn', 'ft', 'face_normals', 'albedo', 'ks', 'bump']
        for attr in attrs:
            value = getattr(self, attr)
            if value is not None:
                setattr(self, attr, value.detach())
        return self

    @classmethod
    def load(cls, path=None, resize=False, auto_uv=True, flip_yz=False, **kwargs):
        # assume init with kwargs
        if path is None:
            mesh = cls(**kwargs)
        # obj supports face uv
        elif path.endswith(".obj"):
            mesh = cls.load_obj(path, **kwargs)
        # trimesh only supports vertex uv, but can load more formats
        else:
            mesh = cls.load_trimesh(path, **kwargs)

        print(f"[Mesh loading] v: {mesh.v.shape}, f: {mesh.f.shape}")

        if resize:
            mesh.auto_size()

        if mesh.vn is None:
            mesh.auto_normal()
        print(f"[Mesh loading] vn: {mesh.vn.shape}, fn: {mesh.fn.shape}")

        if mesh.vt is None and auto_uv:
            mesh.auto_uv(cache_path=path)

        if mesh.vt is not None and mesh.ft is not None:
            print(f"[Mesh loading] vt: {mesh.vt.shape}, ft: {mesh.ft.shape}")

        if flip_yz:
            mesh.v[..., [1, 2]] = mesh.v[..., [2, 1]]
            mesh.vn[..., [1, 2]] = mesh.vn[..., [2, 1]]
            mesh.v[..., 1] = -mesh.v[..., 1]
            mesh.vn[..., 1] = -mesh.vn[..., 1]

        return mesh

    # load from obj file
    @classmethod
    def load_obj(cls, path, albedo_path=None, device=None):
        assert os.path.splitext(path)[-1] == ".obj"

        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # load obj
        with open(path, "r") as f:
            lines = f.readlines()

        def parse_f_v(fv):
            # pass in a vertex term of a face, return {v, vt, vn} (-1 if not provided)
            # supported forms:
            # f v1 v2 v3
            # f v1/vt1 v2/vt2 v3/vt3
            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            # f v1//vn1 v2//vn2 v3//vn3
            xs = [int(x) - 1 if x != "" else -1 for x in fv.split("/")]
            xs.extend([-1] * (3 - len(xs)))
            return xs[0], xs[1], xs[2]

        # NOTE: we ignore usemtl, and assume the mesh ONLY uses one material (first in mtl)
        vertices, texcoords, normals = [], [], []
        faces, tfaces, nfaces = [], [], []
        mtl_path = None

        for line in lines:
            split_line = line.split()
            # empty line
            if len(split_line) == 0:
                continue
            prefix = split_line[0].lower()
            # mtllib
            if prefix == "mtllib":
                mtl_path = split_line[1]
            # v/vn/vt
            elif prefix == "v":
                vertices.append([float(v) for v in split_line[1:]])
            elif prefix == "vn":
                normals.append([float(v) for v in split_line[1:]])
            elif prefix == "vt":
                val = [float(v) for v in split_line[1:]]
                texcoords.append([val[0], 1.0 - val[1]])
            elif prefix == "f":
                vs = split_line[1:]
                nv = len(vs)
                v0, t0, n0 = parse_f_v(vs[0])
                for i in range(nv - 2):  # triangulate (assume vertices are ordered)
                    v1, t1, n1 = parse_f_v(vs[i + 1])
                    v2, t2, n2 = parse_f_v(vs[i + 2])
                    faces.append([v0, v1, v2])
                    tfaces.append([t0, t1, t2])
                    nfaces.append([n0, n1, n2])

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if len(normals) > 0
            else None
        )

        # see if there is vertex color
        if mesh.v.size(-1) > 3:
            mesh.vc = mesh.v[:, 3:]
            mesh.v = mesh.v[:, :3]
            if mesh.vc.size(-1) == 3:
                mesh.vc = torch.cat([mesh.vc, torch.ones_like(mesh.vc[:, :1])], dim=-1)
            print(f"[load_obj] use vertex color: {mesh.vc.shape}")

        # try to retrieve mtl file
        mtl_path_candidates = []
        if mtl_path is not None:
            mtl_path_candidates.append(mtl_path)
            mtl_path_candidates.append(os.path.join(os.path.dirname(path), mtl_path))
        mtl_path_candidates.append(path.replace(".obj", ".mtl"))

        mtl_path = None
        for candidate in mtl_path_candidates:
            if os.path.exists(candidate):
                mtl_path = candidate
                break

        # if albedo_path is not provided, try retrieve it from mtl
        if mtl_path is not None and albedo_path is None:
            with open(mtl_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                split_line = line.split()
                # empty line
                if len(split_line) == 0:
                    continue
                prefix = split_line[0]
                # NOTE: simply use the first map_Kd as albedo!
                if "map_Kd" in prefix:
                    albedo_path = os.path.join(os.path.dirname(path), split_line[1])
                    print(f"[load_obj] use texture from: {albedo_path}")
                    break
                # if "map_ks" in prefix:
                #     ks_path = os.path.join(os.path.dirname())

        # still not found albedo_path, or the path doesn't exist
        if albedo_path is None or not os.path.exists(albedo_path):
            # init an empty texture
            print(f"[load_obj] init empty albedo!")
            # albedo = np.random.rand(1024, 1024, 3).astype(np.float32)

            albedo = np.ones((1024, 1024, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])  # default color

            mesh.textureless = True
        else:
            albedo = mesh.imread_texture(albedo_path)
            print(f"[load_obj] load texture: {albedo.shape}")

        mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)
        mesh.ks = torch.zeros_like(mesh.albedo)
        mesh.bump = torch.zeros_like(mesh.albedo)
        return mesh

    @classmethod
    def load_trimesh(cls, path, device=None):
        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # use trimesh to load glb, assume only has one single RootMesh...
        _data = trimesh.load(path)
        if isinstance(_data, trimesh.Scene):
            mesh_keys = list(_data.geometry.keys())
            assert (
                    len(mesh_keys) == 1
            ), f"{path} contains more than one meshes, not supported!"
            _mesh = _data.geometry[mesh_keys[0]]

        elif isinstance(_data, trimesh.Trimesh):
            _mesh = _data

        else:
            raise NotImplementedError(f"type {type(_data)} not supported!")

        if hasattr(_mesh.visual, "material"):
            _material = _mesh.visual.material
            if isinstance(_material, trimesh.visual.material.PBRMaterial):
                texture = np.array(_material.baseColorTexture).astype(np.float32) / 255
            elif isinstance(_material, trimesh.visual.material.SimpleMaterial):
                texture = (
                        np.array(_material.to_pbr().baseColorTexture).astype(np.float32) / 255
                )
            else:
                raise NotImplementedError(f"material type {type(_material)} not supported!")

            print(f"[load_obj] load texture: {texture.shape}")
            mesh.albedo = torch.tensor(texture, dtype=torch.float32, device=device)

        if hasattr(_mesh.visual, "uv"):
            texcoords = _mesh.visual.uv
            texcoords[:, 1] = 1 - texcoords[:, 1]
            mesh.vt = (
                torch.tensor(texcoords, dtype=torch.float32, device=device)
                if len(texcoords) > 0
                else None
            )
        else:
            texcoords = None

        if hasattr(_mesh.visual, "vertex_colors"):
            colors = _mesh.visual.vertex_colors
            mesh.vc = (
                torch.tensor(colors, dtype=torch.float32, device=device) / 255
                if len(colors) > 0
                else None
            )

        vertices = _mesh.vertices

        normals = _mesh.vertex_normals

        # trimesh only support vertex uv...
        faces = tfaces = nfaces = _mesh.faces

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if texcoords is not None
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if normals is not None
            else None
        )

        return mesh

    # aabb
    def aabb(self):
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self):
        vmin, vmax = self.aabb()
        self.ori_center = (vmax + vmin) / 2
        self.ori_scale = 1.2 / torch.max(vmax - vmin).item()  # to ~ [-0.6, 0.6]
        self.v = (self.v - self.ori_center) * self.ori_scale

    def auto_normal(self):
        i0, i1, i2 = self.f[:, 0].long(), self.f[:, 1].long(), self.f[:, 2].long()
        v0, v1, v2 = self.v[i0, :], self.v[i1, :], self.v[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        face_normals = F.normalize(face_normals, dim=-1)
        vn = torch.zeros_like(self.v)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)
        vn = F.normalize(vn, dim=-1)

        self.vn = vn
        self.fn = self.f
        self.face_normals = face_normals

    def auto_uv(self, cache_path=None, vmap=True):
        # try to load cache
        if cache_path is not None:
            cache_path = os.path.splitext(cache_path)[0] + '_uv.npz'

        if cache_path is not None and os.path.exists(cache_path):
            data = np.load(cache_path)
            vt_np, ft_np, vmapping = data['vt'], data['ft'], data['vmapping']
        else:
            v_np = self.v.detach().cpu().numpy()
            f_np = self.f.detach().int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            # chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # save to cache
            if cache_path is not None:
                np.savez(cache_path, vt=vt_np, ft=ft_np, vmapping=vmapping)

        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)
        self.vt = vt
        self.ft = ft

        if vmap:
            # remap v/f to vt/ft, so each v correspond to a unique vt. (necessary for gltf)
            vmapping = torch.from_numpy(vmapping.astype(np.int64)).long().to(self.device)
            self.align_v_to_vt(vmapping)

    def align_v_to_vt(self, vmapping=None):
        # remap v/f and vn/vn to vt/ft.
        if vmapping is None:
            ft = self.ft.view(-1).long()
            f = self.f.view(-1).long()
            vmapping = torch.zeros(self.vt.shape[0], dtype=torch.long, device=self.device)
            vmapping[ft] = f  # scatter, randomly choose one if index is not unique
        if self.vn is not None and (self.f == self.fn).all():
            self.vn = self.vn[vmapping]
            self.fn = self.ft
        self.v = self.v[vmapping]
        self.f = self.ft

    def align_vn_to_vt(self, vmapping=None):
        if vmapping is None:
            ft = self.ft.view(-1).long()
            fn = self.f.view(-1).long()
            vmapping = torch.zeros(self.vt.shape[0], dtype=torch.long, device=self.device)
            vmapping[ft] = fn  # scatter, randomly choose one if index is not unique
        self.vn = self.vn[vmapping]
        self.fn = self.ft

    def to(self, device):
        self.device = device
        for name in ['v', 'f', 'vn', 'fn', 'vt', 'ft', 'albedo', 'vc', 'face_normals']:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self

    def copy(self):
        return Mesh(
            v=self.v,
            f=self.f,
            vn=self.vn,
            fn=self.fn,
            vt=self.vt,
            ft=self.ft,
            vc=self.vc,
            albedo=self.albedo,
            ks=self.ks,
            bump=self.bump,
            device=self.device,
            textureless=self.textureless)

    def write(self, path, flip_yz=False):
        mesh = self.copy()
        if flip_yz:
            mesh.v = mesh.v.clone()
            mesh.vn = mesh.vn.clone()
            mesh.v[..., 1] = -mesh.v[..., 1]
            mesh.vn[..., 1] = -mesh.vn[..., 1]
            mesh.v[..., [1, 2]] = mesh.v[..., [2, 1]]
            mesh.vn[..., [1, 2]] = mesh.vn[..., [2, 1]]
        if path.endswith('.ply'):
            mesh.write_ply(path)
        elif path.endswith('.obj'):
            mesh.write_obj(path)
        elif path.endswith('.glb') or path.endswith('.gltf'):
            mesh.write_glb(path)
        else:
            raise NotImplementedError(f'format {path} not supported!')

    # write to ply file (only geom)
    def write_ply(self, path):

        v_np = self.v.detach().cpu().numpy()
        f_np = self.f.detach().cpu().numpy()

        _mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
        _mesh.export(path)

    # write to gltf/glb file (geom + texture)
    def write_glb(self, path):

        assert self.vn is not None
        if self.vt is None:
            self.vt = self.v.new_zeros((self.v.size(0), 2))
            self.ft = self.f
        if (self.f != self.ft).any():
            self.align_v_to_vt()
        if (self.fn != self.ft).any():
            self.align_vn_to_vt()

        assert self.v.shape[0] == self.vn.shape[0] and self.v.shape[0] == self.vt.shape[0]

        f_np = self.f.detach().cpu().numpy().astype(np.uint32)
        v_np = self.v.detach().cpu().numpy().astype(np.float32)
        vt_np = self.vt.detach().cpu().numpy().astype(np.float32)
        vn_np = self.vn.detach().cpu().numpy().astype(np.float32)

        albedo = self.albedo.detach().cpu().numpy() if self.albedo is not None \
            else np.full((1024, 1024, 3), 0.5, dtype=np.float32)
        albedo = (albedo * 255).astype(np.uint8)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)

        f_np_blob = f_np.flatten().tobytes()
        v_np_blob = v_np.tobytes()
        vt_np_blob = vt_np.tobytes()
        vn_np_blob = vn_np.tobytes()
        albedo_blob = cv2.imencode('.png', albedo)[1].tobytes()

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[pygltflib.Mesh(primitives=[
                pygltflib.Primitive(
                    # indices to accessors (0 is triangles)
                    attributes=pygltflib.Attributes(
                        POSITION=1, TEXCOORD_0=2, NORMAL=3
                    ),
                    indices=0, material=0,
                )
            ])],
            materials=[
                pygltflib.Material(
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorTexture=pygltflib.TextureInfo(index=0, texCoord=0),
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    ),
                    alphaCutoff=0,
                    doubleSided=True,
                )
            ],
            textures=[
                pygltflib.Texture(sampler=0, source=0),
            ],
            samplers=[
                pygltflib.Sampler(magFilter=pygltflib.LINEAR, minFilter=pygltflib.LINEAR_MIPMAP_LINEAR,
                                  wrapS=pygltflib.REPEAT, wrapT=pygltflib.REPEAT),
            ],
            images=[
                # use embedded (buffer) image
                pygltflib.Image(bufferView=4, mimeType="image/png"),
            ],
            buffers=[
                pygltflib.Buffer(
                    byteLength=len(f_np_blob) + len(v_np_blob) + len(vt_np_blob) + len(vn_np_blob) + len(albedo_blob))
            ],
            # buffer view (based on dtype)
            bufferViews=[
                # triangles; as flatten (element) array
                pygltflib.BufferView(
                    buffer=0,
                    byteLength=len(f_np_blob),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER,  # GL_ELEMENT_ARRAY_BUFFER (34963)
                ),
                # positions; as vec3 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob),
                    byteLength=len(v_np_blob),
                    byteStride=12,  # vec3
                    target=pygltflib.ARRAY_BUFFER,  # GL_ARRAY_BUFFER (34962)
                ),
                # texcoords; as vec2 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob) + len(v_np_blob),
                    byteLength=len(vt_np_blob),
                    byteStride=8,  # vec2
                    target=pygltflib.ARRAY_BUFFER,
                ),
                # normals; as vec3 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob) + len(v_np_blob) + len(vt_np_blob),
                    byteLength=len(vn_np_blob),
                    byteStride=12,  # vec3
                    target=pygltflib.ARRAY_BUFFER,
                ),
                # texture; as none target
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob) + len(v_np_blob) + len(vt_np_blob) + len(vn_np_blob),
                    byteLength=len(albedo_blob),
                ),
            ],
            accessors=[
                # 0 = triangles
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.UNSIGNED_INT,  # GL_UNSIGNED_INT (5125)
                    count=f_np.size,
                    type=pygltflib.SCALAR,
                    max=[int(f_np.max())],
                    min=[int(f_np.min())],
                ),
                # 1 = positions
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,  # GL_FLOAT (5126)
                    count=len(v_np),
                    type=pygltflib.VEC3,
                    max=v_np.max(axis=0).tolist(),
                    min=v_np.min(axis=0).tolist(),
                ),
                # 2 = texcoords
                pygltflib.Accessor(
                    bufferView=2,
                    componentType=pygltflib.FLOAT,
                    count=len(vt_np),
                    type=pygltflib.VEC2,
                    max=vt_np.max(axis=0).tolist(),
                    min=vt_np.min(axis=0).tolist(),
                ),
                # 3 = normals
                pygltflib.Accessor(
                    bufferView=3,
                    componentType=pygltflib.FLOAT,
                    count=len(vn_np),
                    type=pygltflib.VEC3,
                    max=vn_np.max(axis=0).tolist(),
                    min=vn_np.min(axis=0).tolist(),
                ),
            ],
        )

        # set actual data
        gltf.set_binary_blob(f_np_blob + v_np_blob + vt_np_blob + vn_np_blob + albedo_blob)

        # glb = b"".join(gltf.save_to_bytes())
        gltf.save(path)

    @staticmethod
    def write_texture(image, save_path):
        np_image = image.detach().cpu().numpy()
        np_image = (np_image * 255).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))

    @staticmethod
    def imread_texture(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        albedo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = albedo.astype(np.float32) / 255
        return image
    # write to obj file (geom + texture)
    def write_obj(self, path):

        mtl_path = path.replace(".obj", ".mtl")
        albedo_path = os.path.join(os.path.dirname(mtl_path), "texture_kd.png")
        ks_path = os.path.join(os.path.dirname(mtl_path), "texture_ks.png")
        bump_path = os.path.join(os.path.dirname(mtl_path), "texture_kn.png")

        v_np = self.v.detach().cpu().numpy()
        vt_np = self.vt.detach().cpu().numpy() if self.vt is not None else None
        vn_np = self.vn.detach().cpu().numpy() if self.vn is not None else None
        f_np = self.f.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy() if self.ft is not None else None
        fn_np = self.fn.detach().cpu().numpy() if self.fn is not None else None

        with open(path, "w") as fp:
            fp.write(f"mtllib {os.path.basename(mtl_path)} \n")

            for v in v_np:
                fp.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} \n")

            if vt_np is not None:
                for v in vt_np:
                    fp.write(f"vt {v[0]:.4f} {1 - v[1]:.4f} \n")

            if vn_np is not None:
                for v in vn_np:
                    fp.write(f"vn {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} \n")

            fp.write(f"usemtl defaultMat \n")
            for i in range(len(f_np)):
                fp.write("f")
                for j in range(3):
                    fp.write(f' {f_np[i, j] + 1}/{ft_np[i, j] + 1 if ft_np is not None else ""}/{fn_np[i, j] + 1 if fn_np is not None else ""}')
                fp.write("\n")

        with open(mtl_path, "w") as fp:
            fp.write(f"newmtl defaultMat \n")
            fp.write(f"Ka 1 1 1 \n")
            fp.write(f"Kd 1 1 1 \n")
            fp.write(f"Ks 0 0 0 \n")
            fp.write(f"Tr 1 \n")
            fp.write(f"illum 1 \n")
            fp.write(f"Ns 0 \n")

            if not self.textureless and self.albedo is not None:
                fp.write(f"map_Kd {os.path.basename(albedo_path)} \n")
                self.write_texture(self.albedo, albedo_path)

            if not self.textureless and self.ks is not None:
                fp.write(f"map_Ks {os.path.basename(ks_path)} \n")
                self.write_texture(self.ks, ks_path)

            if not self.textureless and self.bump is not None:
                fp.write(f"bump {os.path.basename(bump_path)} \n")
                self.write_texture(self.bump, bump_path)
