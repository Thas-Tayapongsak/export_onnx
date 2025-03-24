from optimum.exporters.onnx.model_configs import ViTOnnxConfig
from optimum.utils import DummyVisionInputGenerator

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
    

class RetinaNetObjectDetectionInputGenerator(DummyVisionInputGenerator):

    SUPPORTED_INPUT_NAMES = (
        "pixel_values",
        "image_sizes"
    )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "image_sizes":
            return self.random_int_tensor(
                shape=[self.batch_size, 2],
                min_value=1,
                max_value=max(self.height, self.width),
                framework=framework,
                dtype=int_dtype,
            )
        
        elif input_name == "pixel_values":
            return self.random_float_tensor(
                shape=[self.batch_size, self.num_channels, self.height, self.width],
                framework=framework,
                dtype=float_dtype,
            )

class RetinaNetOnnxConfig(ViTOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (RetinaNetObjectDetectionInputGenerator,)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "image_sizes": {0: "batch_size", 1: "image_dimensions"}
        }

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