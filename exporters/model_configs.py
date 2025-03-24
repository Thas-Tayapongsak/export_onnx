from optimum.exporters.onnx.model_configs import ViTOnnxConfig

from typing import OrderedDict, Dict

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
                "boxes": {0: "batch_size", 1: "num_predictions", 2: "bbox_coordinates"},
                "scores": {0: "batch_size", 1: "num_predictions"},
                "labels": {0: "batch_size", 1: "num_predictions"},
            }
        )
    
__all__ = [
    'EfficientNetOnnxConfig',
    'MobileNetV3OnnxConfig',
    'RetinaNetOnnxConfig'
]