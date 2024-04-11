# render_utils.py

import math
import numpy as np
import os
import trimesh
from PIL import Image
import cv2
from lib.render.mesh import compute_normal
from lib.render.camera import Camera
from lib.render.gl.geo_render import GeoRender
from lib.render.gl.color_render import ColorRender

def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)
    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def render_obj_files(obj_files, output_dir, width=512, height=512, geo_render=False):
    if geo_render:
        renderer = GeoRender(width=width, height=height)
    else:
        renderer = ColorRender(width=width, height=height)
    cam = Camera(width=1.0, height=height/width)
    cam.ortho_ratio = 1.2
    cam.near = -100
    cam.far = 10

    for obj_path in obj_files:
        obj_file = os.path.basename(obj_path)
        obj_root = os.path.dirname(obj_path)
        file_name = obj_file[:-4]

        if not os.path.exists(obj_path):
            continue    
        mesh = trimesh.load(obj_path)
        vertices = mesh.vertices
        faces = mesh.faces

        R = make_rotate(math.radians(180),0,0)
        bbox_max = vertices.max(0)
        bbox_min = vertices.min(0)
        vertices -= 0.5 * (bbox_max + bbox_min)[None,:]
        vertices /= bbox_max[1] - bbox_min[1]

        normals = compute_normal(vertices, faces)
        
        if geo_render:
            renderer.set_mesh(vertices, faces, normals, faces)
        else:
            renderer.set_mesh(vertices, faces, 0.5*normals+0.5, faces) 
            
        cnt = 0
        for j in range(0, 361, 2):
            cam.center = np.array([0, 0, 0])
            cam.eye = np.array([2.0*math.sin(math.radians(j)), 0, 2.0*math.cos(math.radians(j))]) + cam.center

            renderer.set_camera(cam)
            renderer.display()
            
            img = renderer.get_color(0)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

            img_pil = Image.fromarray((255*img).astype(np.uint8))
            img_pil.save(os.path.join(obj_root, f'rot_{cnt:04d}.png'))

            cnt += 1

    # GIFに結合
    images = []
    for i in range(cnt):
        images.append(Image.open(os.path.join(obj_root, f'rot_{i:04d}.png')))

    gif_path = os.path.join(output_dir, 'rendered.gif')
    images[0].save(gif_path, save_all=True, append_images=images[1:], disposal=2, duration=100, loop=0)
