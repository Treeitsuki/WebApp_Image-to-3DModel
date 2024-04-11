import os
import argparse
from .recon import reconWrapper

def pifuhd_func():
       # PIFuHDの設定
       parser = argparse.ArgumentParser()
       parser.add_argument('-i', '--input_path', type=str, default='./sample_images')
       parser.add_argument('-o', '--out_path', type=str, default='./results')
       parser.add_argument('-c', '--ckpt_path', type=str, default='./checkpoints/pifuhd.pt')
       parser.add_argument('-r', '--resolution', type=int, default=512)
       parser.add_argument('--use_rect', action='store_true', help='use rectangle for cropping')
       args = parser.parse_args()

       # 3D形状の復元
       resolution = str(args.resolution)
       start_id = -1
       end_id = -1
       cmd = ['--dataroot', args.input_path, '--results_path', args.out_path,
              '--loadSize', '1024', '--resolution', resolution, '--load_netMR_checkpoint_path',
              args.ckpt_path,
              '--start_id', '%d' % start_id, '--end_id', '%d' % end_id]
       reconWrapper(cmd, args.use_rect)

       # 生成されたobjファイルのパスを取得
       obj_file = os.path.join(args.out_path, 'pifuhd_final', 'recon', 'result_test_512.obj')

       return obj_file

