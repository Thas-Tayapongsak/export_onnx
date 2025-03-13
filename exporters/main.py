from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, PretrainedConfig
from optimum.exporters.onnx import export, OnnxConfig
from optimum.exporters import TasksManager
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
    
    return model

def create_onnx_config(config: PretrainedConfig, task: str) -> OnnxConfig:
    """
    Parameters
    ----------
    config: AutoConfig
        AutoConfig config of huggingface model
    task: str
        Task of the model

    Returns
    -------
    OnnxConfig:
        onnx config for config's model type and task
    """
    model_type = config.model_type
    if model_type not in onnx_config_constructor_map.keys():
        onnx_config_constructor = TasksManager.get_exporter_config_constructor('onnx', task = task, model_type = model_type)
    else:
        onnx_config_constructor = onnx_config_constructor_map.get(model_type)
    onnx_config = onnx_config_constructor(config, task = task)

    return onnx_config    

def export_onnx(repo_id: str , task: str = 'feature-extraction', output_path: str = '', abs_path: str= ''):
    """
    Parameters
    ----------
    output_path: str
        relative path where model.onnx file will be stored
    abs_path: str
        absolute path where model.onnx file will be stored
    """
    if task not in auto_class_task_map.keys():
        # logger error
        pass

    #check if path exist

    print(repo_id, task, output_path)

    # TODO:
    # test this later
    # output might need rework

    model = create_model(repo_id, task)
    onnx_config = create_onnx_config(model.config, task)
    if abs_path:
        onnx_path = Path(abs_path) / repo_id
    elif output_path:
        onnx_path = Path.cwd() / output_path / repo_id 
    else:
        onnx_path = Path.cwd() / 'onnx' / repo_id 
    onnx_path.mkdir(parents=True, exist_ok=True)
    onnx_inputs, onnx_outputs = export(model, onnx_config, onnx_path / 'model.onnx', onnx_config.DEFAULT_ONNX_OPSET)

    return onnx_inputs, onnx_outputs

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
        '-o','--output-path',
        type=str,
        default='',
        help='relative output path for model.onnx file'
    ),
    parser.add_argument(
        '-a','--abs-path',
        type=str,
        default='',
        help='absolute output path for model.onnx file'
    )
    args = parser.parse_args()
    return args

def main(args):
    export_onnx(args.repo_id, args.task, args.output_path, args.abs_path)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

__all__ = [
    'export_onnx',
    'parse_arguments'
]