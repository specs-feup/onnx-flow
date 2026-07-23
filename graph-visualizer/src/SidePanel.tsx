import { useState } from 'react';
import type { CSSProperties } from 'react';
import { type TransformationOpportunity } from "./api/api.ts"

export default function SidePanel(props: { style: CSSProperties, transformationsOps: TransformationOpportunity[] | null }) {
    return(
        <aside style={props.style}>
            <form>
                {props.transformationsOps?.map((op) => (
                    <input type="radio" id={op.id} name="opportunity">{op.recipeName} - {op.targetNodeId}</input>
                ))    
                }
            </form>
        </aside>
    );
}