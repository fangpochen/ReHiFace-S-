# -- coding: utf-8 --
# @Time : 2022/7/29
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from .base_wrapper import ONNXModel, OnnxModelPickable
from pathlib import Path

try:
    from .base_wrapper import TRTWrapper
except:
    print('trt model needs tensorrt !')


class ModelBase:
    def __init__(self, model_info, provider):
        self.model_path = model_info['model_path']

        if 'input_dynamic_shape' in model_info.keys():
            self.input_dynamic_shape = model_info['input_dynamic_shape']
        else:
            self.input_dynamic_shape = None

        if 'picklable' in model_info.keys():
            picklable = model_info['picklable']
        else:
            picklable = False

        # init model
        if Path(self.model_path).suffix == '.engine':
            self.model_type = 'trt'
            self.model = TRTWrapper(self.model_path)
        else:
            self.model_type = 'onnx'
            if not picklable:
                self.model = ONNXModel(self.model_path, provider=provider, input_dynamic_shape=self.input_dynamic_shape)
            else:
                self.model = OnnxModelPickable(self.model_path, provider=provider, )
