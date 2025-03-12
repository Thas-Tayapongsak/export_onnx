from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
from optimum.exporters.onnx import export
from pathlib import Path

import argparse

from .model_configs import *

onnx_config_constructor_map = {
        'efficientnet': EfficientNetOnnxConfig
    }

auto_class_task_map = {
        'feature-extraction': AutoModel,
        'image-classification': AutoModelForImageClassification
    }

def create_model(repo_id: str, task: str):

    auto_class = auto_class_task_map.get(task)
    model = auto_class.from_pretrained(repo_id, trust_remote_code = True)  

    config = AutoConfig.from_pretrained(repo_id, trust_remote_code = True)
    
    return model, config

def create_onnx_config(config: AutoConfig, task: str):

    model_type = config.model_type
    onnx_config_constructor = onnx_config_constructor_map.get(model_type)
    onnx_config = onnx_config_constructor(config, task = task)

    return onnx_config    

def export_onnx(repo_id: str , task: str = 'feature-extraction', output: str = ''):
    """
    Parameters
    ----------
    output_path : Path
        relative path where model.onnx file will be stored
    """
    if task not in auto_class_task_map.keys():
        # logger error
        pass

    #check if path exist

    print(repo_id, task, output)

    # TODO:
    # test this later
    # output might need rework

    # model, config = create_model(repo_id, task)
    # onnx_config = create_onnx_config(config, task)
    # onnx_path = Path.cwd() / output / repo_id / 'model.onnx'
    # onnx_inputs, onnx_outputs = export(model, onnx_config, onnx_path, onnx_config.DEFAULT_ONNX_OPSET)

    # return onnx_inputs, onnx_outputs

def parse_arguments():
    parser = argparse.ArgumentParser(prog='export_onnx')
    parser.add_argument(
        '-r','--repo-id',
        type=str,
        required=True,
        help='Repo ID on Huggingface'
    )
    parser.add_argument(
        '-t','--task',
        type=str,
        default='feature-extraction',
        help='The model\'s task'
    )
    parser.add_argument(
        '-o','--output',
        type=str,
        default='',
        help='output path for model.onnx file'
    )
    args = parser.parse_args()
    return args

def main(args):
    export_onnx(args.repo_id, args.task, args.output)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

__all__ = [
    'export_onnx',
    'parse_arguments'
]