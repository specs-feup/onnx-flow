export default function EdgePopup({
   selectedEdge,
   popupPos,
   onClose
}: any) {
   if (!selectedEdge || !popupPos) return null;

   return (
     <div>
       <div
         style={{
           position: "fixed",
           inset: 0,
           zIndex: 9998,
         }}
         onClick={onClose}
       />
       
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
         onClick={(e) => e.stopPropagation()}
       >
         <div className="nodee">Edge Details</div>
         <div style={{ marginTop: "5px" }}><b>ID:</b> {selectedEdge.id}</div>
         <div><b>Source:</b> {selectedEdge.source}</div>
         <div><b>Target:</b> {selectedEdge.target}</div>
         { selectedEdge.innerTarget && 
         <div><b>Inner Target:</b> {selectedEdge.innerTarget}</div>
         }
       </div>
     </div>
   );
}