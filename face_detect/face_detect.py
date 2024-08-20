# -- coding: utf-8 --
# @Time : 2021/11/10
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
import cv2
from .scrfd_insightface import SCRFD
import os

def np_norm(x):
    return (x - np.average(x)) / np.std(x)

SCRFD_MODEL_PATH = 'pretrain_models/'


class FaceDetect:
    def __init__(self, mode='scrfd_500m', tracking_thres=0.15):
        self.tracking_thres = tracking_thres
        self.last_bboxes_ = []
        self.dis_list = []
        self.bboxes = self.kpss = self.image = None
        if 'scrfd' in mode:
            scrfd_model_path = SCRFD_MODEL_PATH + 'scrfd_500m_bnkps_shape640x640.onnx'
            self.det_model = SCRFD(scrfd_model_path)
            self.det_model.prepare(ctx_id=0, input_size=(640, 640))
        elif mode == 'mtcnn':
            pass

    def get_bboxes(self, image, nms_thresh=0.5, max_num=0, tracking_init_bbox=None):
        if type(image) == str:
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        elif type(image) == np.ndarray:
            # print('Got np array, assert its cv2 output.')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # tracking logic
        if tracking_init_bbox is None:
            self.last_bboxes_ = None
            self.bboxes, self.kpss = self.det_model.detect(image, thresh=nms_thresh, max_num=max_num,
                                                           metric='default')
            return True, self.bboxes, self.kpss
            # return self.bboxes, self.kpss
        else:
            self.bboxes, self.kpss = self.det_model.detect(image, thresh=nms_thresh, max_num=max_num,
                                                           metric='default')
            if not self.last_bboxes_:
                return self.tracking_filter(tracking_init_bbox)
            else:
                return self.tracking_filter(self.last_bboxes_[0])

    def tracking_filter(self, tracking_init_bbox):
        self.dis_list = []
        for i in range(len(self.bboxes)):
            eye_dis = np.linalg.norm(self.kpss[0][0] - self.kpss[0][1])
            self.dis_list.append(
                np.linalg.norm(np_norm(self.bboxes[i] / eye_dis) - np_norm(tracking_init_bbox / eye_dis)))
            # print(self.dis_list)
        if not self.dis_list or np.min(np.array(self.dis_list)) > self.tracking_thres:
            # print('ok',np.min(np.array(self.dis_list)) )
            self.last_bboxes_ = None
            return False, [], []
        # print(np.min(np.array(self.dis_list)))
        best_index = np.argmin(np.array(self.dis_list))

        self.last_bboxes_ = [self.bboxes[best_index]]
        return True, self.last_bboxes_, [self.kpss[best_index]]

# if __name__ == '__main__':
#
#     fd = FaceDetect()
#     img_path = 'test_img/fake.jpeg'
#     bboxes, kpss = fd.get_bboxes(img_path)
#
#     img = cv2.imread(img_path)
#
#     for i in range(bboxes.shape[0]):
#         bbox = bboxes[i]
#         x1, y1, x2, y2, score = bbox.astype(int)
#         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         if kpss is not None:
#             kps = kpss[i]
#             for kp in kps:
#                 kp = kp.astype(int)
#                 cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)
#     filename = img_path.split('/')[-1]
#     print('output:', filename)
#     cv2.imwrite('./%s' % filename, img)
