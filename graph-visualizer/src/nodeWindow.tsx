export default function NodePopup({selectedNode, popupPos, onClose }) {
  if (!selectedNode || !popupPos) return null;
  return (
    <div className="nodepopup"
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
      <div>Type: {selectedNode["onnxData"].tensorType || selectedNode["onnxData"].opType || "Constant"} <button className="popupBTN" onClick={onClose}>X</button></div>

    </div>
  );
}