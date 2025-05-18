import pickle

from multiprocessing.dummy import Process, Manager, Queue
import cv2
import numpy as np
# close onnxruntime warning
import onnxruntime
onnxruntime.set_default_logger_severity(3)
import os
from face_feature.hifi_image_api import HifiImage
from options.hifi_test_options import HifiTestOptions


class GenInput(Process):
    def __init__(self, feature_src_list_, frame_queue_in_, frame_queue_out_, src_img_path):
        super().__init__()
        self.frame_queue_in = frame_queue_in_
        self.frame_queue_out = frame_queue_out_
        self.feature_src_list = feature_src_list_
        self.src_img_path = src_img_path
        self.hi = HifiImage(crop_size=256)

    def run(self):
        # 从图片直接提取特征
        src_latent, crop_face = self.hi.get_face_feature(self.src_img_path)
        human_feature = [src_latent, crop_face]
        self.feature_src_list.append([human_feature])
        print(f"已加载人脸特征: {self.src_img_path}")

        cap = cv2.VideoCapture(0)  # 640 480  1280 720  1920 1080
        cap.set(3, 1920)
        cap.set(4, 1080)
        print(cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        print('CAP_PROP_FPS',cap.get(cv2.CAP_PROP_FPS))

        count = index = 0
        while True:
            _, frame = cap.read()
            self.frame_queue_in.put(frame)

            count += 1
            # 每500帧换一次脸（如果有多张照片）
            if count % 500 == 0 and os.path.exists(self.src_img_path.replace(".jpg", f"_{index+1}.jpg")):
                next_face_path = self.src_img_path.replace(".jpg", f"_{index+1}.jpg")
                src_latent, crop_face = self.hi.get_face_feature(next_face_path)
                human_feature = [src_latent, crop_face]
                self.feature_src_list.append([human_feature])
                print(f'更换人脸: {next_face_path}')
                index += 1
            if count % 5000 == 0:
                # 退出条件
                self.frame_queue_in.put(None)
                break


class GetOutput(Process):
    def __init__(self, frame_queue_out_):
        super().__init__()
        self.frame_queue_out = frame_queue_out_

    def run(self):
        import time
        count = 0
        fps_count = 0

        start_time = time.time()
        while True:
            queue_out = self.frame_queue_out.get()
            # print(queue_out)
            frame_out = queue_out[0]
            face_detect_flag = queue_out[1]
            # print(face_detect_flag)
            fps_count += 1

            if fps_count % 300 == 0:
                end_time = time.time()
                print('fps: {}'.format(fps_count / (end_time - start_time)))
                start_time = time.time()
                fps_count = 0
            count += 1
            if count % 2500 == 0:
                break
            cv2.imshow('output', frame_out)
            cv2.waitKey(1)


class FaceSwap(Process):
    def __init__(self, feature_src_list_, frame_queue_in_,
                 frame_queue_out_, model_name=''):
        super().__init__()
        from HifiFaceAPI_parallel_trt_roi_realtime_api import HifiFaceRealTime
        self.hfrt = HifiFaceRealTime(feature_src_list_, frame_queue_in_,
                                     frame_queue_out_, model_name=model_name)

    def run(self):
        self.hfrt.forward()


if __name__ == '__main__':
    # 使用与inference.py相同的参数解析方式
    opt = HifiTestOptions().parse()
    
    src_img_path = opt.src_img_path
    model_name = opt.model_name
    
    if not os.path.exists(src_img_path):
        print(f"错误：找不到人脸图片 {src_img_path}")
        print("请使用 --src_img_path 指定正确的人脸图片路径")
        exit(1)
    
    print(f"使用人脸图片: {src_img_path}")
    print(f"使用模型: {model_name}")
    
    frame_queue_in = Queue(2)
    frame_queue_out = Queue(2)
    manager = Manager()
    image_feature_src_list = manager.list()

    gi = GenInput(image_feature_src_list, frame_queue_in, frame_queue_out, src_img_path)
    go = GetOutput(frame_queue_out)
    fs = FaceSwap(image_feature_src_list, frame_queue_in, frame_queue_out, model_name=model_name)

    gi.start()
    go.start()
    fs.start()

    gi.join()
    print('gi stop')
    go.join()
    print('go stop')
    fs.join()
    print('fs stop')

    print('final stop')
