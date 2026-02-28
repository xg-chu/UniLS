"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
Modified from smplx code for FLAME by Xuangeng Chu (xg.chu@outlook.com)
"""

import os

import torch
import torch.nn as nn

from .lbs import lbs


class FLAMEModel(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, n_shape, n_exp, flame_version="2020", with_texture=False):
        super().__init__()
        self.n_shape, self.n_exp = n_shape, n_exp
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        self.flame_path = os.path.join(_abs_path, "../../../assets")
        self.flame_ckpt = torch.load(os.path.join(self.flame_path, f"flame_{flame_version}.pt"), weights_only=True)
        if with_texture:
            self.flame_texture_ckpt = torch.load(os.path.join(self.flame_path, f"flame_texture.pt"), weights_only=True)
            for key in ["verts_uvs", "tex_rgb"]:
                self.flame_texture_ckpt[key] = self.flame_texture_ckpt[key].to(torch.float32)
        else:
            self.flame_texture_ckpt = None
        self.dtype = torch.float32
        self.register_buffer("faces_tensor", self.flame_ckpt["f"])
        self.register_buffer("v_template", self.flame_ckpt["v_template"])
        shapedirs = self.flame_ckpt["shapedirs"]
        self.register_buffer(
            "shapedirs",
            torch.cat([shapedirs[:, :, :n_shape], shapedirs[:, :, 300 : 300 + n_exp]], 2),
        )
        num_pose_basis = self.flame_ckpt["posedirs"].shape[-1]
        self.register_buffer("posedirs", self.flame_ckpt["posedirs"].reshape(-1, num_pose_basis).T)
        self.register_buffer("J_regressor", self.flame_ckpt["J_regressor"])
        parents = self.flame_ckpt["kintree_table"][0]
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("lbs_weights", self.flame_ckpt["weights"])
        # Fixing Eyeball and neck rotation
        self.register_buffer("eye_pose", torch.zeros([1, 6], dtype=torch.float32))
        self.register_buffer("neck_pose", torch.zeros([1, 3], dtype=torch.float32))

        eye_vertices = list(range(4477, 4594, 4)) + [4598, 4602, 4597] + list(range(3931, 4048, 4)) + [4052, 4056, 4051]
        verts_rgb = torch.ones_like(self.v_template) * torch.tensor([142, 179, 247])[None, None]  # (1, V, 3)
        verts_rgb[:, eye_vertices, :] = torch.tensor([21, 60, 122])[None, None].float()
        self.register_buffer("verts_rgb", verts_rgb[0])

    def get_faces(
        self,
    ):
        return self.faces_tensor.long()

    def get_colors(
        self,
    ):
        return self.verts_rgb

    def get_tuv(
        self,
    ):
        return self.flame_texture_ckpt

    def forward(self, shape=None, expression=None, gpose=None, jaw_pose=None, eye_pose=None):
        """
        Input:
            shape: N X number of shape parameters
            expression: N X number of expression parameters
            gpose: N X number of global pose parameters (3)
            jaw_pose: N X number of j parameters (3)
            eye_pose: N X number of eye pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape.shape[0] if shape is not None else expression.shape[0]
        if shape is None:
            shape = self.v_template.new_zeros(batch_size, self.n_shape)
        if expression is None:
            expression = self.v_template.new_zeros(batch_size, self.n_exp)
        if gpose is None:
            gpose = self.v_template.new_zeros(batch_size, 3)
        if jaw_pose is None:
            jaw_pose = self.v_template.new_zeros(batch_size, 3)
        if eye_pose is None:
            eye_pose = self.v_template.new_zeros(batch_size, 6)
        if jaw_pose.shape[1] == 1:
            jaw_pose = torch.cat([jaw_pose, jaw_pose.new_zeros(batch_size, 2)], dim=1)
        if eye_pose.shape[1] == 4:
            eye_zeros = eye_pose.new_zeros(batch_size, 1)
            eye_pose = torch.cat([eye_pose[:, :2], eye_zeros, eye_pose[:, 2:], eye_zeros], dim=1)

        # build flame
        betas = torch.cat([shape, expression], dim=1)
        full_pose = torch.cat([gpose, self.neck_pose.expand(batch_size, -1), jaw_pose, eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, head_joints = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
            detach_pose_correctives=False,
        )
        return vertices

    def get_flame_verts(self, motion_code, shape_code=None, with_headpose=True):
        assert motion_code.dim() == 3
        assert motion_code.shape[-1] == 108
        batch_size, motion_length, _ = motion_code.shape
        exp_code, gpose_code, jaw_code, eyepose_code = motion_code.split([100, 3, 1, 4], dim=-1)
        if not with_headpose:
            gpose_code = torch.zeros_like(gpose_code)
        if shape_code is not None:
            shape_code = shape_code.repeat(1, motion_length, 1)
        else:
            shape_code = [shape_code] * batch_size
        verts = []
        for bidx in range(batch_size):
            this_verts = self.forward(
                shape=shape_code[bidx],
                expression=exp_code[bidx],
                gpose=gpose_code[bidx],
                jaw_pose=jaw_code[bidx],
                eye_pose=eyepose_code[bidx],
            )
            verts.append(this_verts)
        verts = torch.stack(verts, dim=0)
        return verts
