export default function NodePopup({selectedNode, popupPos, onClose }) {
  if (!selectedNode || !popupPos) return null;
  return (
    <div
      style={{
        position: "absolute",
        left:popupPos.x + 20,
        top:popupPos.y + 20,
        background: "rgba(22, 23, 29)",
        color: "#fff",
        padding: "8px",
        border: "3px solid  #fff",
        borderRadius: "5px",
        zIndex:9999,
      }}>
      <div className="nodee">Operation Node</div>
      <div>ID: {selectedNode.id}</div>
      <div>Type: {selectedNode["__specs-onnx__tensor_node"]?.type || selectedNode["__specs-onnx__operation_node"]?.type || "Constant"}</div>
      {selectedNode.label && (<div>Label: {selectedNode.label}</div>)}
      <button className="popupBTN" onClick={onClose}>X</button>
    </div>
  );
}