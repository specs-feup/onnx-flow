import { useRef, useState } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import stylesheet from './styleSheet.ts';

export default function NodePopup({ selectedNode, popupPos, onClose, ...props }: any) {
  const cyRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [, setCyReady] = useState(false);
  const [showLoopWindow, setShowLoopWindow] = useState(false);

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
    <div>
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 9998,
        }}
        onClick={onClose}/>
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
        onClick={(e) => e.stopPropagation()}>
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
      <br></br>
      <button className="loop-btn" onClick={() => setShowLoopWindow(true)}><b>Open Loop</b></button>
      {showLoopWindow && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 10000,
            background: 'rgba(0, 0, 0, 0.48)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onClick={() => setShowLoopWindow(false)}>
          <div
            style={{
              width: '1100px',
              height: '600px',
              background: '#1d1b20',
              border: '2px solid #747474',
              borderRadius: '8px',
              padding: '12px',
            }}
            onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <p><b>Loop Content</b></p>
              <button className="close-btn" onClick={() => setShowLoopWindow(false)}>Exit</button>
            </div>
            <div style={{width:'100%', height:'100%' }} ref={containerRef}>
              {props.cytoscapeData && (
                <CytoscapeComponent
                  elements={CytoscapeComponent.normalizeElements(props.cytoscapeData.elements)}
                  style={{ width: '100%', height: '100%' }}
                  stylesheet={stylesheet as any}
                  layout={props.layout}
                  cy={(cy) => { cyRef.current = cy; setCyReady(true); }}
                />
              )}
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
