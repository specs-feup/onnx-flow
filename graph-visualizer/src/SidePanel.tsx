import { useState } from 'react';
import type { CSSProperties } from 'react';

export default function SidePanel(props: { style: CSSProperties }) {
    return(
        <aside style={props.style}>
            <h1>Hello</h1>
        </aside>
    );
}