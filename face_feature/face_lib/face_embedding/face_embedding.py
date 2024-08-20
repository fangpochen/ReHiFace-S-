# -- coding: utf-8 --
# @Time : 2021/11/10
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from cv2box import CVImage, MyFpsCounter
from cv2box.utils.math import Normalize
from model_lib import ModelBase
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
def down_sample(target_, size):
    import torch.nn.functional as F
    return F.interpolate(target_, size=size, mode='bilinear', align_corners=True)


MODEL_ZOO = {
    'CurricularFace-tjm': {
        'model_path': 'pretrain_models/CurricularFace.tjm',
    }
}

class FaceEmbedding(ModelBase):
    def __init__(self, model_type='CurricularFace-tjm', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        self.input_std = self.input_mean = 127.5
        self.input_size = (112, 112)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.gpu = True if provider=='gpu' else False

    def latent_from_image(self, face_image):
        if type(face_image) == str:
            face_image = cv2.imread(face_image)
            # face_image = cv2.resize(face_image, (224, 224))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image = Image.fromarray(face_image)
        elif type(face_image) == np.ndarray:
            face_image = Image.fromarray(face_image)
            # print('got np array, assert its cv2 output.')
        with torch.no_grad():
            face = self.transformer(face_image)
            face = face.unsqueeze(0)
            if self.gpu:
                face = face.cuda()
            # 输入尺寸为(112, 112)  RGB
            face = down_sample(face, size=[112, 112])
            # 人脸latent code为512维
            face_latent = self.model(face)
            face_latent = F.normalize(face_latent, p=2, dim=1)
        return face_latent[0]

if __name__ == '__main__':
    # CurricularFace
    fb_cur = FaceEmbedding(model_type='CurricularFace-tjm', provider='gpu')
    latent_cur = fb_cur.latent_from_image('data/source/ym-1.jpeg')
    print(latent_cur.shape)
    print(latent_cur)