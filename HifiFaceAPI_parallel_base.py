import os
import time
import numpy as np

import numexpr as ne
# ne.set_num_threads(10)

from multiprocessing.dummy import Process, Queue
from face_detect.face_align_68 import face_alignment_landmark
from face_detect.face_detect import FaceDetect
from face_lib.face_swap import HifiFace
from face_restore.gfpgan_onnx_api import GFPGAN
from face_restore.xseg_onnx_api import XSEG

TRACKING_THRESHOLD = 0.15

# def np_norm(x):
#     return (x - np.average(x)) / np.std(x)

def cosine_vectorized_v3(array1, array2):
    sumyy = np.einsum('ij,ij->i', array2, array2)
    sumxx = np.einsum('ij,ij->i', array1, array1)[:, None]
    sumxy = array1.dot(array2.T)
    sqrt_sumxx = ne.evaluate('sqrt(sumxx)')
    sqrt_sumyy = ne.evaluate('sqrt(sumyy)')
    return ne.evaluate('(sumxy/sqrt_sumxx)/sqrt_sumyy')


class Consumer0Base(Process):
    def __init__(self, opt, frame_queue_in, feature_dst_list=None, queue_list=None, block=True, fps_counter=False):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.pid = os.getpid()

        self.opt = opt
        self.frame_queue_in = frame_queue_in
        self.feature_dst_list = feature_dst_list
        self.crop_size = self.opt.input_size
        self.scrfd_detector = FaceDetect(mode='scrfd_500m', tracking_thres=TRACKING_THRESHOLD)
        self.face_alignment = face_alignment_landmark(lm_type=68)

        print('init consumer {}, pid is {}.'.format(self.__class__.__name__, self.pid))


class Consumer1BaseONNX(Process):
    def __init__(self, opt, feature_list, queue_list: list, block=True, fps_counter=False,provider='gpu', load_xseg=True, xseg_flag=False):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.pid = os.getpid()
        self.opt = opt
        self.feature_list = feature_list
        # self.index_list = index_list
        # self.apply_gpen = apply_gpen
        self.crop_size = self.opt.input_size
        self.xseg_flag = xseg_flag

        print("model_name:", self.opt.model_name)
        self.hf = HifiFace(model_name='er8_bs1', provider=provider)
        if load_xseg:
            self.xseg = XSEG(model_type='xseg_0611', provider=provider)

    def switch_xseg(self):
        self.xseg_flag = not self.xseg_flag

    def predict(self, src_face_image, dst_face_latent):
        mask_out, swap_face_out = self.hf.forward(src_face_image, dst_face_latent)
        if self.xseg_flag:
            mask_out = self.xseg.forward(swap_face_out)[None,None]
        return [mask_out, swap_face_out]


class Consumer2Base(Process):
    def __init__(self, queue_list: list, frame_queue_out, block=True, fps_counter=False):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.pid = os.getpid()
        self.frame_queue_out = frame_queue_out

        # from face_restore import FaceRestore
        # self.fa = FaceRestore(use_gpu=True, mode='gfpgan')  # gfpgan gpen dfdnet

        print('init consumer {}, pid is {}.'.format(self.__class__.__name__, self.pid))

    def run(self):
        counter = 0
        start_time = time.time()

        while True:
            something_in = self.queue_list[0].get()

            # exit condition
            if something_in is None:
                print('subprocess {} exit !'.format(self.pid))
                break

            self.forward_func(something_in)

            if self.fps_counter:
                counter += 1
                if (time.time() - start_time) > 4:
                    print("Consumer2 FPS: {}".format(counter / (time.time() - start_time)))
                    counter = 0
                    start_time = time.time()
        print('c2 stop')
        # cv2.destroyAllWindows()

class Consumer3Base(Process):
    def __init__(self, queue_list, block=True, fps_counter=False, provider='gpu'):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.pid = os.getpid()

        self.gfp = GFPGAN(model_type='GFPGANv1.4', provider=provider)

        print('init consumer {}, pid is {}.'.format(self.__class__.__name__, self.pid))

    def run(self):
        counter = 0
        start_time = time.time()

        while True:
            something_in = self.queue_list[0].get()

            if something_in is None:
                print('subprocess {} exit !'.format(self.pid))
                self.queue_list[1].put(None)
                break

            self.forward_func(something_in)


            if self.fps_counter:
                counter += 1
                if (time.time() - start_time) > 4:
                    print("Consumer3 FPS: {}".format(counter / (time.time() - start_time)))
                    counter = 0
                    start_time = time.time()

        print('c3 stop')

