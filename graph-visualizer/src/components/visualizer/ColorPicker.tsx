/**
 * @file ColorPicker.tsx
 * @description Color picker component providing an interactive HTML5 color input
 * for customizing node background tinting across the Cytoscape graph.
 */

/**
 * Properties for the ColorPicker component.
 */
interface ColorPickerProps {
    /** Current hex color string value (defaults to '#533b6e') */
    value?: string;
    /** Callback fired when a new color is picked */
    onChange?: (v: string) => void;
}

/**
 * Color picker button control for graph node styling.
 *
 * @param props - ColorPicker properties
 * @returns JSX element containing the color picker button
 */
export default function ColorPicker({ value, onChange }: ColorPickerProps) {
    return (
        <button className="button-group">Node Color
            <input
                type="color"
                value={value || '#533b6e'}
                onChange={(node) => onChange?.(node.target.value)}
                style={{ width: '24px', height: '24px', padding: "0px", border: 'none', background: 'transparent' }}
            />
        </button>
    );
}