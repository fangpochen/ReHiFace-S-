# -- coding: utf-8 --
# @Time : 2022/7/29
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from .base_wrapper import ONNXModel, OnnxModelPickable
from pathlib import Path
import torch

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

        if 'trt_wrapper_self' in model_info.keys():
            TRTWrapper = TRTWrapperSelf

        # init model
        if Path(self.model_path).suffix == '.engine':
            self.model_type = 'trt'
            self.model = TRTWrapper(self.model_path)
        elif Path(self.model_path).suffix == '.tjm':
            self.model_type = 'tjm'
            self.model =torch.jit.load(self.model_path)
            self.model.eval()
        elif Path(self.model_path).suffix in ['.onnx', '.bin']:
            self.model_type = 'onnx'
            model_name = self.model_path.split('/')[-1].split('.')[0].split('_')[0]
            if not picklable:
                self.model = ONNXModel(self.model_path, provider=provider, input_dynamic_shape=self.input_dynamic_shape, model_name=model_name)
            else:
                self.model = OnnxModelPickable(self.model_path, provider=provider, )
        else:
            raise 'check model suffix , support engine/tjm/onnx now.'
