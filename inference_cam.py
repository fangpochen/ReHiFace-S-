import pickle

from multiprocessing.dummy import Process, Manager, Queue
import cv2
import numpy as np
# close onnxruntime warning
import onnxruntime
onnxruntime.set_default_logger_severity(3)


class GenInput(Process):
    def __init__(self, feature_src_list_, frame_queue_in_, frame_queue_out_):
        super().__init__()
        self.frame_queue_in = frame_queue_in_
        self.frame_queue_out = frame_queue_out_
        self.feature_src_list = feature_src_list_

    def run(self):
        with open('data/image_feature_dict.pkl', 'rb') as f:
            image_feature_src_dict = pickle.load(f)

        print(len(image_feature_src_dict))
        self.feature_src_list.append([image_feature_src_dict['1']])

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
            if count % 500 == 0:
                self.feature_src_list.append([image_feature_src_dict['{}'.format(1 + index)],
                                              image_feature_src_dict['{}'.format(10 + index)]])
                print('change src face')
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
    frame_queue_in = Queue(2)
    frame_queue_out = Queue(2)
    manager = Manager()
    image_feature_src_list = manager.list()

    gi = GenInput(image_feature_src_list, frame_queue_in, frame_queue_out)
    go = GetOutput(frame_queue_out)
    fs = FaceSwap(image_feature_src_list, frame_queue_in, frame_queue_out, model_name='er8_bs1')

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
