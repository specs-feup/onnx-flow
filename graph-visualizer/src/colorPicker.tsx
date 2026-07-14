import React from 'react';

export default function ColorPicker({ value,onChange }: { value?: string;
    onChange?: (v:string) => void }) {
    return (
        <button className="button-group">Node Color
            <input
                type="color"
                value={value || '#533b6e'}
                onChange={(node) => onChange?.(node.target.value)}
                style={{width: '24px', height: '24px', padding:"0px", border: 'none',background: 'transparent'}}
            />
        </button>
    );
}