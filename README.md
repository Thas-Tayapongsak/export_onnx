# export_onnx

For model types not yet supported by optimum

## Python

```
project_folder
|_ export_onnx
|_ example.py
```

In example.py

```python
  from export_onnx import *

  repo_id = "example/my_model"
  task = "image-classification"
  export_onnx(repo_id, task)
```

Will create 

```
project_folder
|_ export_onnx
|_ onnx
|  |_ example
|    |_ my_model
|      |_ model.onnx
|_ example.py
```

## CMD

```
project_folder
|_ export_onnx
```

Calling 

```shell
C:\Users\example\project_folder> python export_onnx -r 'example/my_model' -t 'image-classification'
```

Gives

```
project_folder
|_ export_onnx
|_ onnx
  |_ example
    |_ my_model
      |_ model.onnx
```
