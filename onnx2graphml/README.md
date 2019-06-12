# ONNX2GraphML

Utility for converting ONNX files into the GraphML format for viewing, analysis and grammar mining.

## Requirements
Requires Python 2.7 and the libraries listed in `requirements.txt`

## Install
Run the following commands from the command-line.

```bash
git clone https://github.com/mike-holcomb/onnx2graphml.git
cd onnx2graphml
python setup.py install --user
```

## Example
An example Jupyter notebook `example_convert.ipynb` is included here for reference as well for library usage.  In summary:

```python2
from onnx2graph import convert
convert.convert('model.onnx','model.graphml')
```

A minimal command-line tool is also included as well:

```bash
python onnx2graphml.py --onnx_file="mlp.onnx" --graphml_file="mlp.graphml"
```

Additional help information:
```
python onnx2graphml.py --help

       USAGE: onnx2graphml.py [flags]
flags:

onnx2graphml.py:
  --graphml_file: Destination for GraphML model file output
    (default: 'model.graphml')
  --onnx_file: ONNX model file input
```


## Acknowledgements
This code borrows heavily from the graph traversal code of https://github.com/onnx/onnx/blob/master/onnx/helper.py from the ONNX Project Contributors.

## TODO
* Clean up the code/remove unnecessary comments
* Add edge attributes for data dimensionality
* Incorporate model parameter inputs
