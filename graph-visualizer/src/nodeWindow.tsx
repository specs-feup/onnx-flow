export default function NodePopup({ selectedNode, popupPos, onClose }) {
  if (!selectedNode || !popupPos) return null;
  var typeNode;
  if (selectedNode.onnxData.tensorType) {
    typeNode = "Tensor";
  } else if (selectedNode.onnxData.opType) {
    typeNode = "Operation";
  } else {
    typeNode = "Constant";
  }
  return (
    <div
      className="nodepopup"
      style={{
        position: "absolute",
        left: popupPos.x + 20,
        top: popupPos.y + 20,
        background: "rgba(22, 23, 29)",
        color: "#fff",
        padding: "8px",
        border: "2px solid #747474",
        borderRadius: "5px",
        zIndex: 9999,
      }}
    >
      <div className="nodee">{typeNode} Node</div>
      <div>ID: {selectedNode.id}</div>
      <div>Type: {selectedNode["onnxData"].tensorType || selectedNode["onnxData"].opType || "Constant"}</div>

      {selectedNode.onnxData.tensorType && (<div>Literal Type: {selectedNode.onnxData.literalType}</div>)}
      {selectedNode.onnxData.tensorType && (<div>Shape: {selectedNode.onnxData.shape.join(", ")}</div>)}
      {selectedNode.onnxData.tensorType && selectedNode.onnxData.shape.metadata && JSON.stringify(selectedNode.onnxData.shape.metadata)!='{}' &&(<div>Metadata: {selectedNode.onnxData.shape.metadata}</div>)}

      {selectedNode.onnxData.opType && JSON.stringify(selectedNode.onnxData.attributes)!='{}' && (<div>Attributes: {JSON.stringify(selectedNode.onnxData.attributes)}</div>)}
      {selectedNode.onnxData.opType && (<div>Inputs: {selectedNode.onnxData.inputs.join(", ")}</div>)}
      {selectedNode.onnxData.opType && selectedNode.onnxData.inputs.metadata && JSON.stringify(selectedNode.onnxData.inputs.metadata)!='{}' &&(<div>Metadata: {selectedNode.onnxData.inputs.metadata}</div>)}

      {!selectedNode.onnxData.tensorType && !selectedNode.onnxData.opType && (<div>Data Type: {selectedNode.onnxData.proto.dataType}</div>)} 
      <button className="popupBTN" onClick={onClose}>x</button>
    </div>
  );
}
