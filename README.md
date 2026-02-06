# @specs-feup/onnx-flow

**A high-performance tool for decomposing and optimizing ONNX models into hardware-aware data-flow representations.**

`onnx-flow` transforms high-level neural network operations (like `AveragePool`, `Conv`, or `Softmax`) into lower-level operations (suitable for offloading). It enables decompositions, explicit loop-lowering, optimizations (such as loop fusion), and graph partitioning for distributed workloads. Outputs can be generated in ONNX, JSON and DOT formats with multiple visualization options.

---

## Key Features

* **Operation Decomposition**: Transforms high-level neural network operations (like `AveragePool`, `Conv`, or `Softmax`) into lower-level primitives suitable for offloading.
* **Optimizations**: Applies advanced optimizations, such as loop fusion, to streamline execution.
* **Graph Partitioning**: Supports splitting graphs into partitions to facilitate distributed workloads, automatically managing boundary tensors.
* **Multi-Format Output**: Generates processed graphs in ONNX, JSON, and DOT formats to suit various integration needs.
* **Visualization**: Provides multiple visualization options, including static SVG rendering and interactive Graphviz Online links.

---

## Installation

To install the package via npm:

```bash
npm install @specs-feup/onnx-flow
```

---

## CLI Usage

The CLI is the primary way to transform models. It supports both `.onnx` binaries and `.json` flow-graph exports.

```bash
onnx-flow <input_file> [options]
```

### 1. Partitioning Options
*Overrides transformation options to ensure split-point stability.*
* `--partition, --pt <nodeId | OpType Instance>`: Partition the graph into head/tail at a specific node (e.g., `--pt 12` or `--pt MatMul 2`).

### 2. Transformation & Optimization
* `-f, --fuse`: Fuse supported operators into a single Loop (Default: `true`).
* `-c, --coalesce`: Use coalesced scalar MAC for `MatMul` inside Loop bodies (Default: `true`).
* `-r, --recurse`: Recursively decompose generated loop bodies (Default: `false`).
* `--ll, --loopLowering`: Enable explicit Loop node generation (Default: `true`).
* `--dgc, --decomposeForCgra`: Apply CGRA-specific decomposition logic.

### 3. Output & Visualization
* `-o, --output <path>`: Save the resulting graph to a specific file.
* `--fm, --format <json|dot>`: Choose output format, besides reconverted ONNX (Default: `json`).
* `--vz, --visualization <0|1|2>`: `0` = None, `1` = Graphviz Online link, `2` = Local server.
* `--fmtr, --formatter <default|cgra>`: Choose the DOT styling engine.

### 4. Other
* `--version`: Show version number.
* `--help`: Show detailed usage.

---

## License

Licensed under the **Apache 2.0 License**.