from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
#from apps.test_module import pifuhd_func
from apps.recon import reconWrapper
import shutil

import math
import cv2
from PIL import Image
from lib.render.mesh import compute_normal
from lib.render.camera import Camera
from lib.render.gl.geo_render import GeoRender
from lib.render.gl.color_render import ColorRender
import trimesh
from typing import Union


app = FastAPI()

# CORSを許可する起源の設定
origins = [
    "http://localhost:5173",  # Reactアプリがローカルで実行されている場合
    "http://localhost",
    "http://localhost:8080",
    # その他必要なオリジンを追加
]

# CORSMiddlewareを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

#PIFuHDで3Dデータの生成
@app.post("/pifuhd")
async def pifuhd_func(request: Request):
    # HTTPリクエストからデータを取得
    form_data = await request.form()

    # PIFuHDの設定
    input_path = form_data.get('input_path', './sample_images')
    out_path = form_data.get('out_path', './results')
    ckpt_path = form_data.get('ckpt_path', './checkpoints/pifuhd.pt')
    resolution = int(form_data.get('resolution', 512))
    use_rect = form_data.get('use_rect', False)

    # 3D形状の復元
    resolution = str(resolution)
    start_id = -1
    end_id = -1
    cmd = ['--dataroot', input_path, '--results_path', out_path,
           '--loadSize', '1024', '--resolution', resolution, '--load_netMR_checkpoint_path',
           ckpt_path,
           '--start_id', '%d' % start_id, '--end_id', '%d' % end_id]
    reconWrapper(cmd, use_rect)

    # 生成されたobjファイルのパス
    obj_file = os.path.join(out_path, 'pifuhd_final', 'recon', 'result_test_512.obj')

    return JSONResponse(content={"obj_file": obj_file})

@app.post("/pifuhd_save_image")
async def pifuhd_save_image_func(file: UploadFile = File(...), out_path: str = './results'):
    # 送られてきた画像を保存
    file_path = os.path.join(out_path, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # PIFuHDの設定
    ckpt_path = './checkpoints/pifuhd.pt'
    resolution = 512

    # 3D形状の復元
    start_id = -1
    end_id = -1
    cmd = ['--dataroot', out_path, '--results_path', out_path,
           '--loadSize', '1024', '--resolution', str(resolution), '--load_netMR_checkpoint_path',
           ckpt_path,
           '--start_id', '%d' % start_id, '--end_id', '%d' % end_id]
    reconWrapper(cmd, False)

    # 生成されたobjファイルのパス
    obj_file = os.path.join(out_path, 'pifuhd_final', 'recon', 'result_test_512.obj')

    return JSONResponse(content={"obj_file": obj_file})

@app.post("/render_test")
async def render_test(request: Request):        
    gif_path = "/results/pifuhd_final/render/rendered_test.gif"
    return {"GIF_path":  gif_path}

@app.post("/test")
async def obj_test(request: Request):
    obj_path = "./results/pifuhd_final/render/rendered_test.gif"
    return {"OBJ_path": obj_path}

@app.post("/test_save_image")
async def test_save_image(file: UploadFile = File(...), out_path: str = './sample_images'):
    # 送られてきた画像を保存
    file_path = os.path.join(out_path, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    gif_path = '../images/rendered_test.gif'
    
    return JSONResponse(content={"GIF_path": gif_path})

@app.post("/test_gif")
async def process_image(file: UploadFile = File(...)):
    GIF_FILE_PATH = "./results/pifuhd_final/render/rendered_test.gif"
    
    return FileResponse(GIF_FILE_PATH, media_type="image/gif")

#--------------------------------------------------------------------------------------------------------------
#objファイルをGIFへレンダリング

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

@app.post("/render")
async def render(request: Request, file_dir: str = None, geo_render: bool = False):
    # HTTPリクエストからデータを取得
    form_data = await request.form()

    if geo_render:
        renderer = GeoRender(width=width, height=height)
    else:
        renderer = ColorRender(width=width, height=height)
    cam = Camera(width=1.0, height=height/width)
    cam.ortho_ratio = 1.2
    cam.near = -100
    cam.far = 10

    render_root = './results/pifuhd_final/render'
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
        img_pil.save(os.path.join(render_root, 'rot_%04d.png' % cnt))

        cnt += 1
        
    # GIFに結合
    images = []
    for i in range(cnt):
        images.append(Image.open(os.path.join(render_root, 'rot_%04d.png' % i)))
    
    front_images_path = '../React_tailwind/src/images'
    
    gif_path = os.path.join(render_root, 'rendered_test.gif')
    images[0].save(gif_path, save_all=True, append_images=images[1:], disposal=2, duration=50, loop=0)
    
    # 保存していた画像を削除
    for i in range(cnt):
        os.remove(os.path.join(render_root, 'rot_%04d.png' % i))
    
    return {"GIF_path":  gif_path}


@app.post("/test_pifuhd_and_render")
async def pifuhd_and_render(file: UploadFile = File(...), out_path: str = './results'):
    # 送られてきた画像を保存
    file_path = os.path.join("sample_images", "test.png")
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # PIFuHDの設定
    ckpt_path = './checkpoints/pifuhd.pt'
    resolution = 512

    # 3D形状の復元
    start_id = -1
    end_id = -1
    cmd = ['--dataroot', 'sample_images', '--results_path', out_path,
           '--loadSize', '1024', '--resolution', str(resolution), '--load_netMR_checkpoint_path',
           ckpt_path,
           '--start_id', '%d' % start_id, '--end_id', '%d' % end_id]
    reconWrapper(cmd, False)

    geo_render: bool = False

    if geo_render:
        renderer = GeoRender(width=width, height=height)
    else:
        renderer = ColorRender(width=width, height=height)
    cam = Camera(width=1.0, height=height/width)
    cam.ortho_ratio = 1.2
    cam.near = -100
    cam.far = 10


    render_root = './results/pifuhd_final/render'
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
        img_pil.save(os.path.join(render_root, 'rot_%04d.png' % cnt))

        cnt += 1
        
    # GIFに結合
    images = []
    for i in range(cnt):
        images.append(Image.open(os.path.join(render_root, 'rot_%04d.png' % i)))
    
    gif_path = os.path.join(render_root, 'rendered_test.gif')
    images[0].save(gif_path, save_all=True, append_images=images[1:], disposal=2, duration=50, loop=0)
    
    # 保存していた画像を削除
    for i in range(cnt):
        os.remove(os.path.join(render_root, 'rot_%04d.png' % i))
    
    return FileResponse(gif_path, media_type="image/gif")