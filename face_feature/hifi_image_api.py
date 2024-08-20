# import os
# import sys
# path = os.path.dirname(__file__)
# sys.path.append(path)
from face_feature.face_lib.face_landmark.pfpld import PFPLD
from face_feature.face_lib.face_embedding import FaceEmbedding
from face_feature.face_lib.face_detect_and_align import FaceDetect5Landmarks
import cv2
import numpy as np
from cv2box import CVImage
from PIL import Image
class HifiImage:
    def __init__(self, crop_size=256):
        """
        :param crop_size: 输出字典中展示图片的size
        """
        self.crop_size = crop_size

        self.fe = FaceEmbedding(model_type='CurricularFace-tjm', provider='gpu')
        self.scrfd_detector = FaceDetect5Landmarks(mode='scrfd_500m')
        self.pfpld = PFPLD()

        self.image_feature_dict = {}


    def get_face_feature(self, image_path):
        if isinstance(image_path, str):
            src_image = CVImage(image_path).rgb()
        else:
            src_image = np.array(image_path)
        try:
            borderpad = int(np.max([np.max(src_image.shape[:2]) * 0.1, 100]))
            src_image = np.pad(src_image, ((borderpad, borderpad), (borderpad, borderpad), (0, 0)), 'constant',
                               constant_values=(0, 0))
        except Exception as e:
            print(f'padding fail , got {e}')
            return None
        bboxes_scrfd, kpss_scrfd = self.scrfd_detector.get_bboxes(src_image, min_bbox_size=64)
        image_face_crop_list, m_ = self.scrfd_detector.get_multi_face(crop_size=self.crop_size,
                                                                      mode='mtcnn_256')

        img = np.array(image_face_crop_list[0])
        lm = self.pfpld.forward(img)
        lm[0][5][0] = np.min([lm[0][5][0], lm[0][48][0] - 5])
        lm[0][14][0] = np.max([lm[0][14][0], lm[0][54][0] + 5])

        img = cv2.rectangle(img, lm[0][11].ravel().astype(int), lm[0][14].ravel().astype(int), (0, 0, 0), -1)
        img = cv2.rectangle(img, lm[0][2].ravel().astype(int), lm[0][5].ravel().astype(int), (0, 0, 0), -1)

        assert len(image_face_crop_list) == 1, 'only support single face in input image'
        image_latent = self.fe.latent_from_image(img).cpu().numpy()
        # image_latent = self.fe.forward(img)
        crop_face = image_face_crop_list[0]
        return image_latent, crop_face
