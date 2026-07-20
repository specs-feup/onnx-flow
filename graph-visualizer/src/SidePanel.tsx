import { useState } from 'react';
import type { CSSProperties } from 'react';

export default function SidePanel(props: { style: CSSProperties, selectedNode: any | null }) {
    return(
        <aside style={props.style}>
            {props.selectedNode && 
            <ul>
                <li>Node ID: {props.selectedNode.id}</li>
                <li>Type: {props.selectedNode["onnxData"].tensorType || props.selectedNode["onnxData"].opType || "Constant"}</li>
            </ul>
            }
        </aside>
    );
}