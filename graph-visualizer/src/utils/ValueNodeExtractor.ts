export function valueNodeExtractor(cytoscapeData: any) {
    return cytoscapeData?.elements.nodes.filter((node:any) => {
        const kind = node.data.onnxData.kind;
        return kind === "TensorNode" || 
        kind === "ConstantNode" || 
        kind === "RegionArgumentNode";
    })
}