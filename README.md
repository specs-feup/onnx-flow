# ONNX2Cytoscape
Tool to convert an ONNX graph into a Cytoscape graph, using Cytoscape.js and decompose its high-level operations into low-level operations.
The resulting graph maintains its initial structure. That is (as in all ONNX graphs), the nodes represent operations. The graph as a whole takes inputs and returns an output.

```
Usage: onnx2cytoscape <input_file> [options]

Options:
      --version                Show version number                      [boolean]
  -o, --output                 Output resulting graph to a file         [string]
  -f, --format                 Output format (json or dot)              [string] [choices: "json", "dot"] [default: "json"]
  -v, --verbosity              Control verbosity (0 = silent, 1 = normal/outputs, 2 = verbose)                          [number] [default: 1]
      --noLowLevel, --nl       Disable the low-level conversion         [boolean] [default: false]
      --noOptimize, --no       Disable optimization steps               [boolean] [default: false]
      --noCodegen, --nc        Disable code generation step             [boolean] [default: false]
      --visualization, --vz    Choose visualization option (0 = none, 1 = Graphviz online link, 2 = Graphviz server)    [number] [default: 2]
      --help                   Show help                                [boolean]

You need to provide an input file (ONNX or JSON)
```

