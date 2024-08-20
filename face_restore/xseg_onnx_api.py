# -- coding: utf-8 --
# @Time : 2022/11/8


from cv2box import CVImage, MyFpsCounter

from model_lib import ModelBase
import numpy as np
import cv2

MODEL_ZOO = {
    'xseg_0611': {
        'model_path': './pretrain_models/xseg_230611_16_17.onnx',
        'input_dynamic_shape': [[1, 256, 256, 3]]
    },
}


class XSEG(ModelBase):
    def __init__(self, model_type='xseg_0611', provider='cpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type


    def forward(self, face_image):
        """
        Args:
            face_image: cv2 image -1~1 RGB
        Returns:
            RGB 256x256x3 -1~1
        """
        face_image = (face_image + 1) / 2
        if face_image.shape[-1] >= 4:
            if len(face_image.shape)>3:
                face_image = face_image[0]
            face_image = face_image.transpose(1, 2, 0)
        face_image_h, face_image_w, _ = face_image.shape
        if face_image_h != 256:
            face_image = cv2.resize(face_image, (256, 256))
        image_out = self.model.forward(face_image[...,::-1][None].astype(np.float32))
        # print(image_out[0][0].shape)
        output_face = (image_out[0].squeeze()).clip(0, 1)
        if face_image_h != 256:
            output_face = cv2.resize(output_face, (face_image_w, face_image_h))
        return output_face


if __name__ == '__main__':
    face_img_p = 'data/source/ym-1.jpeg'
    fa = XSEG(model_type='xseg_0611', provider='trt16')
    face_img = (cv2.resize(cv2.imread(face_img_p)/127.5-1,(512,512)))[...,::-1]

    with MyFpsCounter() as mfc:
        for i in range(20):
            face = fa.forward(face_img)
    # CVImage(face, image_format='cv2').save('./xseg.jpg')
    #CVImage(face, image_format='cv2').show()
