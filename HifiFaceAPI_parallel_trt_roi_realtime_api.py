import os
import cv2
import time
import numpy as np
import numexpr as ne
from multiprocessing.dummy import Process, Queue
from options.hifi_test_options import HifiTestOptions
from HifiFaceAPI_parallel_base import Consumer0Base, Consumer2Base, Consumer1BaseONNX


def np_norm(x):
    return (x - np.average(x)) / np.std(x)


def reverse2wholeimage_hifi_trt_roi(swaped_img, mat_rev, img_mask, frame, roi_img, roi_box):
    target_image = cv2.warpAffine(swaped_img, mat_rev, roi_img.shape[:2][::-1], borderMode=cv2.BORDER_REPLICATE)[
                   ...,
                   ::-1]

    local_dict = {
        'img_mask': img_mask,
        'target_image': target_image,
        'roi_img': roi_img,
    }
    img = ne.evaluate('img_mask * (target_image * 255)+(1 - img_mask) * roi_img', local_dict=local_dict,
                      global_dict=None)
    img = img.astype(np.uint8)
    frame[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = img
    return frame


def get_max_face(np_rois):
    roi_areas = []
    for index in range(np_rois.shape[0]):
        roi_areas.append((np_rois[index, 2] - np_rois[index, 0]) * (np_rois[index, 3] - np_rois[index, 1]))
    return np.argmax(np.array(roi_areas))


class Consumer0(Consumer0Base):
    def __init__(self, opt, frame_queue_in, queue_list: list, block=True, fps_counter=False):
        super().__init__(opt, frame_queue_in, None, queue_list, block, fps_counter)

    def run(self):
        counter = 0
        start_time = time.time()
        kpss_old = None
        rois_old = faces_old = Ms_old = masks_old = None

        while True:
            frame = self.frame_queue_in.get()
            if frame is None:
                break
            try:
                _, bboxes, kpss = self.scrfd_detector.get_bboxes(frame, max_num=0)
                rois, faces, Ms, masks = self.face_alignment.forward(
                    frame, bboxes, kpss, limit=5, min_face_size=30,
                    crop_size=(self.crop_size, self.crop_size), apply_roi=True
                )

            except (TypeError, IndexError, ValueError) as e:
                self.queue_list[0].put([None, frame])
                continue

            if len(faces)==0:
                self.queue_list[0].put([None, frame])
                continue
            elif len(faces)==1:
                face = np.array(faces[0])
                mat = Ms[0]
                roi_box = rois[0]
            else:
                max_index = get_max_face(np.array(rois))
                face = np.array(faces[max_index])
                mat = Ms[max_index]
                roi_box = rois[max_index]
            roi_img = frame[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]

           # "The default normalization to the range of -1 to 1, where the model input is in RGB format
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            self.queue_list[0].put([face, mat, [], frame, roi_img, roi_box])

            if self.fps_counter:
                counter += 1
                if (time.time() - start_time) > 10:
                    print("Consumer0 FPS: {}".format(counter / (time.time() - start_time)))
                    counter = 0
                    start_time = time.time()
        self.queue_list[0].put(None)
        print('co stop')


class Consumer1(Consumer1BaseONNX):
    def __init__(self, opt, feature_list, queue_list: list, block=True, fps_counter=False):
        super().__init__(opt, feature_list, queue_list, block, fps_counter)

    def run(self):
        counter = 0
        start_time = time.time()

        while True:
            something_in = self.queue_list[0].get()
            if something_in is None:
                break
            elif len(something_in) == 2:
                self.queue_list[1].put([None, something_in[1]])
                continue


            if len(self.feature_list) > 1:
                self.feature_list.pop(0)

            image_latent = self.feature_list[0][0]

            mask_out, swap_face_out = self.predict(something_in[0], image_latent[0].reshape(1, -1))

            mask = cv2.warpAffine(mask_out[0][0].astype(np.float32), something_in[1],
                                  something_in[4].shape[:2][::-1])
            mask[mask > 0.2] = 1
            mask = mask[:, :, np.newaxis].astype(np.uint8)
            swap_face = swap_face_out[0].transpose((1, 2, 0)).astype(np.float32)

            self.queue_list[1].put(
                [swap_face, something_in[1], mask, something_in[3], something_in[4], something_in[5]])

            if self.fps_counter:
                counter += 1
                if (time.time() - start_time) > 10:
                    print("Consumer1 FPS: {}".format(counter / (time.time() - start_time)))
                    counter = 0
                    start_time = time.time()
        self.queue_list[1].put(None)
        print('c1 stop')


class Consumer2(Consumer2Base):
    def __init__(self, queue_list: list, frame_queue_out, block=True, fps_counter=False):
        super().__init__(queue_list, frame_queue_out, block, fps_counter)
        self.face_detect_flag = True

    def forward_func(self, something_in):

        # do your work here.
        if len(something_in) == 2:
            self.face_detect_flag = False
            frame = something_in[1]
            frame_out = frame.astype(np.uint8)
        else:
            self.face_detect_flag = True
            # swap_face = something_in[0]
            swap_face = ((something_in[0] + 1) / 2)
            frame_out = reverse2wholeimage_hifi_trt_roi(
                swap_face, something_in[1], something_in[2],
                something_in[3], something_in[4], something_in[5]
            )
        self.frame_queue_out.put([frame_out, self.face_detect_flag])
        # cv2.imshow('output', frame_out)
        # cv2.waitKey(1)


class HifiFaceRealTime:

    def __init__(self, feature_dict_list_, frame_queue_in, frame_queue_out, gpu=True, model_name=''):
        self.opt = HifiTestOptions().parse()
        if model_name != '':
            self.opt.model_name = model_name
        self.opt.input_size = 256
        self.feature_dict_list = feature_dict_list_
        self.frame_queue_in = frame_queue_in
        self.frame_queue_out = frame_queue_out

        self.gpu = gpu

    def forward(self):
        self.q0 = Queue(2)
        self.q1 = Queue(2)

        self.c0 = Consumer0(self.opt, self.frame_queue_in, [self.q0], fps_counter=False)
        self.c1 = Consumer1(self.opt, self.feature_dict_list, [self.q0, self.q1], fps_counter=False)
        self.c2 = Consumer2([self.q1], self.frame_queue_out, fps_counter=False)

        self.c0.start()
        self.c1.start()
        self.c2.start()

        self.c0.join()
        self.c1.join()
        self.c2.join()
        return
