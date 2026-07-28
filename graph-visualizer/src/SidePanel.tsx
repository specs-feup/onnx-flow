import type { CSSProperties } from 'react';
import { applyTransformation, fetchGraph, fetchTransformationOpportunities, type TransformationOpportunity} from './api/api.ts';
import type { CytoscapeData } from './Cytoscape.tsx';

export default function SidePanel(props: { 
    style: CSSProperties;
    transformationOps: {
        ops: TransformationOpportunity[],
        setOps: (transformationOps: TransformationOpportunity[]) => void,
    };
    setCytoscapeData: (data: CytoscapeData | null) => void,
    transformationsHistory: {
        undo: {stack: string[], setStack: (history: string[]) => void}
        redo: {stack: string[], setStack: (history: string[]) => void}
    },
}) {
    return(
        <aside style={props.style}>
                {props.transformationOps.ops.map((op) => (
                    <button onClick={ async () => {
                        const operationId: string = op.id;
                        props.transformationOps.setOps([]);
                        props.setCytoscapeData(null);
                        await applyTransformation(3000, operationId);
                        props.transformationsHistory.undo.setStack([...props.transformationsHistory.undo.stack, operationId]);
                        props.transformationsHistory.redo.setStack([]);
                        props.setCytoscapeData(await fetchGraph(3000));
                        props.transformationOps.setOps(await fetchTransformationOpportunities(3000));
                    }  
                    }>{op.recipeName} - {op.targetNodeId}</button>
                ))    
                }
        </aside>
    );
}