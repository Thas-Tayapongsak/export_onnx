from typing import OrderedDict, Dict

from optimum.exporters.onnx.model_configs import ViTOnnxConfig
from optimum.utils import DummyVisionInputGenerator


class EfficientNetOnnxConfig(ViTOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs 

        if self.task == "image-classification":
            common_outputs["logits"] = {0: "batch_size", 1: "num_classes"}
        
        return common_outputs
    

class MobileNetV3OnnxConfig(ViTOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs 

        if self.task == "image-classification":
            common_outputs["logits"] = {0: "batch_size", 1: "num_classes"}
        
        return common_outputs
    

class RetinaNetOnnxConfig(ViTOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return OrderedDict(
            {
                "logits": {0: "batch_size", 1: "num_preds", 2: "num_classes"},
                "pred_boxes": {0: "batch_size", 1: "num_preds", 2: "4"},
                "anchors": {0: "batch_size", 1: "num_anchors", 2: "4"},
                "num_anchors_per_level": {0: "num_levels"},
            }
        )


__all__ = [
    'EfficientNetOnnxConfig',
    'MobileNetV3OnnxConfig',
    'RetinaNetOnnxConfig'
]