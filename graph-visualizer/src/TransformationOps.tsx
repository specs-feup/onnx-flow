import type { CSSProperties } from "react";
import { type TransformationOpportunity, applyTransformation, fetchGraph, fetchTransformationOpportunities } from "./api/api";
import type { CytoscapeData } from "./Cytoscape";
import { useParams } from "react-router-dom";

export default function TransformationOps(
    props: { 
        style: CSSProperties;
        transformationOps: {
            ops: TransformationOpportunity[],
            setOps: (transformationOps: TransformationOpportunity[]) => void,
        };
        setCytoscapeData: (data: CytoscapeData | null) => void,
        transformationsHistory: {
            undo: {stack: string[], setStack: (history: string[]) => void}
            redo: {stack: string[], setStack: (history: string[]) => void}
        };
    }
) {

    const { sessionId } = useParams();

    return(
        <aside style={props.style}>
            {props.transformationOps.ops.map((op) => (
                <button onClick={ async () => {
                    const operationId: string = op.id;
                    props.transformationOps.setOps([]);
                    props.setCytoscapeData(null);
                    await applyTransformation(3000, sessionId!,operationId);
                    props.transformationsHistory.undo.setStack([...props.transformationsHistory.undo.stack, operationId]);
                    props.transformationsHistory.redo.setStack([]);
                    props.setCytoscapeData(await fetchGraph(3000, sessionId!));
                    props.transformationOps.setOps(await fetchTransformationOpportunities(3000, sessionId!));
                }  
                }>{op.recipeName} - {op.targetNodeId}</button>
            ))    
            }
        </aside>
    );
}