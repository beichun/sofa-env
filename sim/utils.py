import os
import time
import collections
import math

import numpy as np
import imageio
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

import Sofa.SofaGL

def init_display(display_size):
    # if pygame.display.get_init:
    #     pygame.display.quit()
    #     pygame.quit()
    # else:
    #     pygame.display.init()
    pygame.display.init()
    pygame.display.set_mode(display_size, pygame.DOUBLEBUF | pygame.OPENGL)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    Sofa.SofaGL.glewInit()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display_size[0] / display_size[1]), 0.1, 1000.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
def render(root):
    """
    Get the OpenGL Context to render an image (snapshot) of the simulation state
    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    cameraMVM = root.camera.getOpenGLModelViewMatrix()
    glMultMatrixd(cameraMVM)
    
    Sofa.SofaGL.draw(root)

    pygame.display.get_surface().fill((0,0,0))
    pygame.display.flip()
    
def get_tableau_palette(alpha=False):
    if alpha:
        palette = np.array([[ 78,121,167, 255], # blue
                            [255, 87, 89, 255], # red
                            [ 89,169, 79, 255], # green
                            [242,142, 43, 255], # orange
                            [237,201, 72, 255], # yellow
                            [176,122,161, 255], # purple
                            [255,157,167, 255], # pink 
                            [118,183,178, 255], # cyan
                            [156,117, 95, 255], # brown
                            [186,176,172, 255]  # gray
                            ],dtype=np.uint8)
    else:
        palette = np.array([[ 78,121,167], # blue
                            [255, 87, 89], # red
                            [ 89,169, 79], # green
                            [242,142, 43], # orange
                            [237,201, 72], # yellow
                            [176,122,161], # purple
                            [255,157,167], # pink 
                            [118,183,178], # cyan
                            [156,117, 95], # brown
                            [186,176,172]  # gray
                            ],dtype=np.uint8)
    return palette/255.0

def load_obj(bullet_client, scaling, position, orientation, visual_mesh, collision_mesh, color, fixed_base=False):
    template = """<?xml version="1.0" encoding="UTF-8"?>
<robot name="obj.urdf">
    <link name="baseLink">
        <contact>
            <lateral_friction value="1.0"/>
            <rolling_friction value="0.0001"/>
            <inertia_scaling value="3.0"/>
        </contact>
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value=".1"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
        <visual>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{0}" scale="1 1 1"/>
            </geometry>
            <material name="Cyan">
                <color rgba="{2} {3} {4} {5}"/>
            </material>
        </visual>
        <collision>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{1}" scale="1 1 1"/>
            </geometry>
        </collision>
    </link>
</robot>"""
    urdf_path = '.tmp_my_obj_%.8f%.8f.urdf' % (time.time(), np.random.rand())

    with open(urdf_path, "w") as f:
        f.write(template.format(visual_mesh, collision_mesh, color[0], color[1], color[2], color[3]))
    body_id = bullet_client.loadURDF(
        fileName=urdf_path,
        basePosition=position,
        baseOrientation=orientation,
        globalScaling=scaling,
        useFixedBase=fixed_base
    )
    os.remove(urdf_path)

    return body_id

def imretype(im, dtype):
    """
    Image retype
    :param im: original image. dtype support: float, float16, float32, float64, uint8, uint16
    :param dtype: target dtype. dtype support: float, float16, float32, float64, uint8, uint16
    
    :return image of new dtype
    """
    im = np.array(im)

    if im.dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(np.float)
    elif im.dtype == 'uint8':
        im = im.astype(np.float) / 255.
    elif im.dtype == 'uint16':
        im = im.astype(np.float) / 65535.
    else:
        raise NotImplementedError('unsupported source dtype: {0}'.format(im.dtype))
    try:
        assert np.min(im) >= 0 and np.max(im) <= 1
    except:
        im = np.clip(im, 0, 1.0)

    if dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(dtype)
    elif dtype == 'uint8':
        im = (im * 255.).astype(dtype)
    elif dtype == 'uint16':
        im = (im * 65535.).astype(dtype)
    else:
        raise NotImplementedError('unsupported target dtype: {0}'.format(dtype))

    return im

def imwrite(path, obj):
    """
    Save Image
    :param path: path to save the image. Suffix support: png or jpg or gif
    :param image: array or list of array(list of image --> save as gif). Shape support: WxHx3 or WxHx1 or 3xWxH or 1xWxH
    """
    if not isinstance(obj, (collections.Sequence, collections.UserList)):
        obj = [obj]
    writer = imageio.get_writer(path)
    for im in obj:
        im = imretype(im, dtype='uint8').squeeze()
        if len(im.shape) == 3 and im.shape[0] == 3:
            im = np.transpose(im, (1, 2, 0))
        writer.append_data(im)
    writer.close()
    

# Get 3D pointcloud from RGB-D image
# Input:
#   color_img - HxWx3 uint8 array of color values in range 0-255
#   depth_img - HxW float array of depth values in meters aligned with color_img
#   segmentation_mask - HxW int array of segmentation instance
#   cam_intr  - 3x3 float array of camera intrinsic parameters
# Output:
#   cam_pts   - Nx3 float array of 3D points in camera coordinates
#   color_pts - Nx3 uint8 array of color values in range 0-255 corresponding to cam_pts
#   segmentation_pts - Nx1 int array of segmentation instance corresponding to cam_pts
def get_pointcloud(color_img, depth_img, segmentation_mask, cam_intr):

    img_h = depth_img.shape[0]
    img_w = depth_img.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,img_w-1,img_w),
                                  np.linspace(0,img_h-1,img_h))
    cam_pts_x = np.multiply(pixel_x-cam_intr[0,2],depth_img/cam_intr[0,0])
    cam_pts_y = np.multiply(pixel_y-cam_intr[1,2],depth_img/cam_intr[1,1])
    cam_pts_z = depth_img
    cam_pts = np.array([cam_pts_x,cam_pts_y,cam_pts_z]).transpose(1,2,0).reshape(-1,3)

    color_pts = None if color_img is None else color_img.reshape(-1,3)
    segmentation_pts = None if segmentation_mask is None else segmentation_mask.reshape(-1, 1)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_img.flatten() > 0)[0]
    cam_pts = cam_pts[valid_depth_ind,:]
    color_pts = None if color_img is None else color_pts[valid_depth_ind,:]
    segmentation_pts = None if segmentation_mask is None else segmentation_pts[valid_depth_ind,:]

    return cam_pts, color_pts, segmentation_pts


# Apply rigid transformation to 3D pointcloud
# Input:
#   xyz_pts      - Nx3 float array of 3D points
#   rigid_transform - 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)
# Output:
#   xyz_pts      - Nx3 float array of transformed 3D points
def transform_pointcloud(xyz_pts,rigid_transform):
    xyz_pts = np.dot(rigid_transform[:3,:3],xyz_pts.T) # apply rotation
    xyz_pts = xyz_pts+np.tile(rigid_transform[:3,3].reshape(3,1),(1,xyz_pts.shape[1])) # apply translation
    return xyz_pts.T


# Get top-down (along z-axis) orthographic heightmap image from 3D pointcloud
# Input:
#   cam_pts          - Nx3 float array of 3D points in world coordinates
#   color_pts        - Nx3 uint8 array of color values in range 0-255 corresponding to cam_pts
#   segmentation_pts - Nx1 int array of segmentation instance corresponding to cam_pts
#   view_bounds      - 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining region in 3D space of heightmap in world coordinates
#   heightmap_pix_sz - float value defining size of each pixel in meters (determines heightmap resolution)
#   zero_level       - float value defining z coordinate of zero level (i.e. bottom) of heightmap 
# Output:
#   depth_heightmap  - HxW float array of height values (from zero level) in meters
#   color_heightmap  - HxWx3 uint8 array of backprojected color values in range 0-255 aligned with depth_heightmap
#   segmentation_heightmap - HxW int array of segmentation instance aligned with depth_heightmap
def get_heightmap(cam_pts,color_pts,segmentation_pts,view_bounds,heightmap_pix_sz,zero_level):

    heightmap_size = np.round(((view_bounds[1,1]-view_bounds[1,0])/heightmap_pix_sz,
                               (view_bounds[0,1]-view_bounds[0,0])/heightmap_pix_sz)).astype(int)

    # Remove points outside workspace bounds
    heightmap_valid_ind = np.logical_and(np.logical_and(
                          np.logical_and(np.logical_and(cam_pts[:,0] >= view_bounds[0,0],
                                                        cam_pts[:,0] <  view_bounds[0,1]),
                                                        cam_pts[:,1] >= view_bounds[1,0]),
                                                        cam_pts[:,1] <  view_bounds[1,1]),
                                                        cam_pts[:,2] <  view_bounds[2,1])
    cam_pts = cam_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]
    segmentation_pts = segmentation_pts[heightmap_valid_ind]

    # Sort points by z value (works in tandem with array assignment to ensure heightmap uses points with highest z values)
    sort_z_ind = np.argsort(cam_pts[:,2])
    cam_pts = cam_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]
    segmentation_pts = segmentation_pts[sort_z_ind]

    # Backproject 3D pointcloud onto heightmap
    heightmap_pix_x = np.floor((cam_pts[:,0]-view_bounds[0,0])/heightmap_pix_sz).astype(int)
    heightmap_pix_y = np.floor((cam_pts[:,1]-view_bounds[1,0])/heightmap_pix_sz).astype(int)

    # Get height values from z values minus zero level
    depth_heightmap = np.zeros(heightmap_size)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = cam_pts[:,2]
    depth_heightmap = depth_heightmap-zero_level
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -zero_level] = 0

    # Map colors
    color_heightmap = np.zeros((heightmap_size[0],heightmap_size[1],3),dtype=np.uint8)
    for c in range(3):
        color_heightmap[heightmap_pix_y,heightmap_pix_x,c] = color_pts[:,c]
    
    # Map segmentations
    segmentation_heightmap = np.zeros((heightmap_size[0],heightmap_size[1]),dtype=np.int)
    segmentation_heightmap[heightmap_pix_y,heightmap_pix_x] = segmentation_pts[:, 0]

    return color_heightmap, depth_heightmap, segmentation_heightmap


class OBJ(object):
    def __init__(self, vertices=None, faces=None, vnormals=None, normal_normalized=True, normalize=False, file_name=None, scale=1.0, rotation_degree=0):
        self.vertices = None
        self.faces = None
        self.vnormals = None
        self.fnormals = None
        self.fareas = None
        self.normalize = normalize

        if file_name is not None:
            self.vertices, self.faces, self.vnormals = obj_loader(file_name, self.normalize)[:3]
        else:
            self.vertices = vertices
            self.faces = faces
            self.vnormals = vnormals
            self.obj_upward_only = None

        if abs(scale - 1.0) > 0.0000001:
            self.vertices *= scale

        if abs(rotation_degree - 0.0) > 0.0001:
            new_x = self.vertices[:, 0]*math.cos(rotation_degree / 180.0 * np.pi) + self.vertices[:, 1]*math.sin(rotation_degree / 180.0 * np.pi)
            new_y = self.vertices[:, 0]*math.sin(rotation_degree / 180.0 * np.pi) - self.vertices[:, 1]*math.cos(rotation_degree / 180.0 * np.pi)
            self.vertices = np.vstack([new_x, new_y*-1.0, self.vertices[:, 2]]).T

        self.num_points = len(self.vertices)
        self.num_faces = len(self.faces)

        if normal_normalized:
            self.cal_vnormals()

    def centers(self):
        return np.mean(self.vertices, axis=0)

    def cal_fareas(self):
        self.fareas = 0.5 * np.linalg.norm(np.cross(self.vertices[self.faces[:, 0], :] - self.vertices[self.faces[:, 1], :], self.vertices[self.faces[:, 0], :] - self.vertices[self.faces[:, 2], :]), axis=1)
        return np.copy(self.fareas)

    def cal_fnormals(self):
        N = np.cross(self.vertices[self.faces[:, 0], :] - self.vertices[self.faces[:, 1], :], self.vertices[self.faces[:, 0], :] - self.vertices[self.faces[:, 2], :])
        row_norms = np.linalg.norm(N, axis=1)
        self.fareas = 0.5 * row_norms
        with np.errstate(divide='ignore',invalid='ignore'):
            N = (N.T / row_norms).T
        self.fnormals = N

    def cal_vnormals(self, force=False):
        if force or self.vnormals is None or (self.vnormals.shape != self.vertices.shape):
            N = np.cross(self.vertices[self.faces[:, 0], :] - self.vertices[self.faces[:, 1], :], self.vertices[self.faces[:, 0], :] - self.vertices[self.faces[:, 2], :])
            self.vnormals = np.zeros_like(self.vertices)
            self.vnormals[self.faces[:, 0], :] += N
            self.vnormals[self.faces[:, 1], :] += N
            self.vnormals[self.faces[:, 2], :] += N
        row_norms = np.linalg.norm(self.vnormals, axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            self.vnormals = (self.vnormals.T / row_norms).T

    def write(self, file_name):
        faces = self.faces + 1
        with open(file_name, 'w') as fw:
            for v in self.vertices:
                fw.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        if self.vnormals.shape == self.vertices.shape:
            for vn in self.vnormals:
                fw.write('vn %f %f %f\n' % (vn[0], vn[1], vn[2]))

        for f in faces:
            fw.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def sample_points(self, n_samples, seed=None, with_normal=False, upward_only=False):
        # if seed is not None:
        # np.random.seed(seed)
        # else:
        # np.random.seed(self.seed)
        if upward_only:
            if self.obj_upward_only is None:
                if self.fnormals is None:
                    self.cal_fnormals()
                face_idx = np.where(self.fnormals[:, -1] >= 0)[0]
                faces = np.copy(self.faces)[face_idx]
                self.obj_upward_only = OBJ(np.copy(self.vertices), faces)
            return self.obj_upward_only.sample_points(n_samples, seed, with_normal)

        face_areas = self.cal_fareas()
        face_areas = face_areas / np.sum(face_areas)

        n_samples_per_face = np.round(n_samples * face_areas)

        n_samples_per_face = n_samples_per_face.astype(np.int)
        n_samples_s = int(np.sum(n_samples_per_face))

        diff = n_samples_s - n_samples
        indices = np.arange(self.num_faces)

        if diff > 0:
            rand_faces = np.random.choice(indices[n_samples_per_face >= 1], abs(diff), replace=False)
            n_samples_per_face[rand_faces] = n_samples_per_face[rand_faces] - 1
        elif diff < 0:
            rand_faces = np.random.choice(indices, abs(diff), replace=False)
            n_samples_per_face[rand_faces] = n_samples_per_face[rand_faces] + 1

        # Create a vector that contains the face indices
        sample_face_idx = np.zeros((n_samples, ), dtype=int)

        acc = 0
        for face_idx, _n_sample in enumerate(n_samples_per_face):
            sample_face_idx[acc:acc + _n_sample] = face_idx
            acc += _n_sample

        r = np.random.rand(n_samples, 2)

        A = self.vertices[self.faces[sample_face_idx, 0], :]
        B = self.vertices[self.faces[sample_face_idx, 1], :]
        C = self.vertices[self.faces[sample_face_idx, 2], :]
        P = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + np.sqrt(r[:, 0:1]) * r[:, 1:] * C

        if with_normal:
            self.cal_vnormals()
            An = self.vnormals[self.faces[sample_face_idx, 0], :]
            Bn = self.vnormals[self.faces[sample_face_idx, 1], :]
            Cn = self.vnormals[self.faces[sample_face_idx, 2], :]

            An[np.isnan(An)] = 0.0
            Bn[np.isnan(Bn)] = 0.0
            Cn[np.isnan(Cn)] = 0.0

            Pn = (1 - np.sqrt(r[:, 0:1])) * An + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * Bn + np.sqrt(r[:, 0:1]) * r[:, 1:] * Cn
            row_norms = np.linalg.norm(Pn, axis=1)
            row_norms[row_norms == 0] = 1
            Pn = (Pn.T / row_norms).T
            return P, Pn, sample_face_idx
        else:
            return P, sample_face_idx
