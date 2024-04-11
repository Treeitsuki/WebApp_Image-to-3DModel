import math
import numpy as np
import sys
import os
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lib.render.mesh import load_obj_mesh, compute_normal
from lib.render.camera import Camera
from lib.render.gl.geo_render import GeoRender
from lib.render.gl.color_render import ColorRender
import trimesh

import cv2
import os
import argparse
from PIL import Image

width = 512
height = 512

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

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_dir', type=str, required=False)
parser.add_argument('-ww', '--width', type=int, default=512)
parser.add_argument('-hh', '--height', type=int, default=512)
parser.add_argument('-g', '--geo_render', action='store_true', help='default is normal rendering')

args = parser.parse_args()

if args.geo_render:
    renderer = GeoRender(width=args.width, height=args.height)
else:
    renderer = ColorRender(width=args.width, height=args.height)
cam = Camera(width=1.0, height=args.height/args.width)
cam.ortho_ratio = 1.2
cam.near = -100
cam.far = 10


obj_root = './results/pifuhd_final/recon'
obj_file = './results/pifuhd_final/recon/result_test_512.obj'

mesh = trimesh.load(obj_file)
vertices = mesh.vertices
faces = mesh.faces

# vertices = np.matmul(vertices, R.T)
bbox_max = vertices.max(0)
bbox_min = vertices.min(0)

# notice that original scale is discarded to render with the same size
vertices -= 0.5 * (bbox_max + bbox_min)[None,:]
vertices /= bbox_max[1] - bbox_min[1]

normals = compute_normal(vertices, faces)
    
if args.geo_render:
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
    img_pil.save(os.path.join(obj_root, 'rot_%04d.png' % cnt))

    cnt += 1
    
# GIFに結合
images = []
for i in range(cnt):
    images.append(Image.open(os.path.join(obj_root, 'rot_%04d.png' % i)))

gif_path = os.path.join(obj_root, 'rendered_test.gif')
images[0].save(gif_path, save_all=True, append_images=images[1:], disposal=2, duration=50, loop=0)