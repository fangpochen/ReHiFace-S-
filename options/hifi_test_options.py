import argparse


class HifiTestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--model_name', type=str, default='er8_bs1', help='er8_bs1')
        self.parser.add_argument('--input_size', type=int, default='256')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self.parser.add_argument('--src_img_path', type=str, default='data/source/elon-musk1.jpg')
        self.parser.add_argument('--video_path', type=str, default='data/source/demo.mp4')
        self.parser.add_argument('--video_to_1080p', action="store_true", help='change video resolution to 1080p')
        self.parser.add_argument('--mode', type=str, default='default', help='default merge')
        self.parser.add_argument('--align_method', type=str, default='68', help='face align method:68 5class')

        self.parser.add_argument('--use_gfpgan', action="store_true", help='use gfpgan for sr or not')
        self.parser.add_argument('--sr_weight', type=float, default=1.0)

        self.parser.add_argument('--use_color_trans', action="store_true", help='use color transfer or not')
        self.parser.add_argument('--color_trans_mode', type=str, default='rct', help='rct lct mkl idt sot')

        self.parser.add_argument('--output_dir', type=str, default='data/output')


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        return self.opt
