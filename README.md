# onnx-flow
Tool to convert an ONNX graph into a data-flow graph, decomposing its high-level operations into low-level operations and performing a set of optimizations. The resulting graph maintains its initial structure, that is (as in all ONNX graphs), the nodes represent operations, initial inputs, and final outputs.

## Installation

To install the package:

```bash
npm install @specs-feup/onnx-flow
```

## CLI Usage

```
Usage: onnx-flow <input_file> [options]

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

## Programmatic Usage

In addition to the CLI, `onnx-flow` can be used programmatically by importing its functions in your project. This allows you to parse ONNX files, manipulate data-flow graphs, and generate outputs programmatically. The available functions are the following:

### `onnxFileParser`
Parses an ONNX file or JSON graph into an ONNX object.
 - Input:
    - Path of input file (`string`)
 - Output: ONNX graph parsed into a JSON file (`json`)

```typescript
import { onnxFileParser } from "@specs-feup/onnx-flow";

const onnxObject = await onnxFileParser("path/to/file.onnx");
console.log(onnxObject);
```

---

### `loadGraph`
Loads an ONNX object into a data-flow graph and optionally applies low-level transformations and optimizations.
 - Input:
    - ONNX graph parsed into a JSON file (`json`)
    - Enable low-level operation decomposition (`boolean`)
    - Enable optimizations (`boolean`)
    - Convert output to DOT format (`boolean`)
 - Output: Resulting flow graph, either the object or in DOT format depending on the option chosen (`flow graph` or `string`)

```typescript
import { loadGraph } from "@specs-feup/onnx-flow";

const dotGraph = loadGraph(onnxObject, true, true, true); // Enable both low-level and optimization steps and convert output to DOT format
console.log(dotGraph);
```

---

### `renderDotToSVG`
Converts a DOT graph string into an SVG string for rendering or embedding.
 - Input:
    - Source graph in DOT format (`string`)
 - Output: SVG image of the graph (`string`)

```typescript
import { renderDotToSVG } from "@specs-feup/onnx-flow";

const dotGraph = "digraph { a -> b }";
const svgContent = await renderDotToSVG(dotGraph);
console.log(svgContent);
```

---

### `generateGraphvizOnlineLink`
Generates a link to visualize a DOT graph on [Graphviz Online](https://dreampuf.github.io/GraphvizOnline/).
 - Input:
    - Source graph in DOT format (`string`)
- Output: Link to open the given graph in Graphviz Online (`string`)

```typescript
import { generateGraphvizOnlineLink } from "@specs-feup/onnx-flow";

const dotGraph = "digraph { a -> b }";
const link = generateGraphvizOnlineLink(dotGraph);
console.log(link); // Outputs: https://dreampuf.github.io/GraphvizOnline/#...
```

---

### `generateGraphCode`
Generates code from a data-flow graph.
 - Input:
    - Source flow graph (with low-level decomposition applied) (`flow graph`)
 - Output: Code corresponding to the source graph (`string`)

```typescript
import { generateGraphCode } from "@specs-feup/onnx-flow";

const code = generateGraphCode(graph);
console.log(code);
```

---

## License

Licensed under the Apache 2.0 License.



