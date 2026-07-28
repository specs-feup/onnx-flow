import { useState } from "react";

export default function NodeAdder() {
    const [nodeKind, setNodeKind] = useState<string>("constant");

    return (
        <aside>
            <label htmlFor="kind">Node Kind: </label>
            <select id="kind" onChange={(e) => setNodeKind(e.target.value)}>
                <option value="constant">Constant Node</option>
                <option value="tensor">Tensor Node</option>
                <option value="operation">Operation Node</option>
            </select>

            {nodeKind === 'operation' && (
                <>
                <label>Operation Type:</label>
                </>
            )}
        </aside>
    );
}
