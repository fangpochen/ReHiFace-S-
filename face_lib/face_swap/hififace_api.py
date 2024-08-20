# -- coding: utf-8 --
# @Time : 2022/8/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
from model_lib import ModelBase

MODEL_ZOO = {
    'er8_bs1': {
        'model_path': 'pretrain_models/9O_865k.onnx',
    },
}


class HifiFace(ModelBase):
    def __init__(self, model_name='er8_bs1', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)

    def forward(self, src_face_image, dst_face_latent):
        """
        Args:
            src_face_image:
            dst_face_latent:
        Returns:
        """
        img_tensor = ((src_face_image.transpose(2, 0, 1) / 255.0) * 2 - 1)[None]
        blob = [img_tensor.astype(np.float32), dst_face_latent.astype(np.float32)]
        output = self.model.forward(blob)
        # print("-------------model_type:",self.model_type)
        if self.model_type == 'trt':
            mask, swap_face = output
        else:
            swap_face, mask = output

        return mask, swap_face
