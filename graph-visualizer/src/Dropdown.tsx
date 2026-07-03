import React, { useState } from 'react';

const Dropdown = () => {
    const [isOpen, setIsOpen] = useState(false);

    const toggleDropdown = () => {
        setIsOpen(!isOpen);
    };
    return (
        <div className="">
                <button
                    onClick={toggleDropdown}
                    className="">
                    Graph Layout
                </button>
            {isOpen && (
                

                <select>
                    <option value="dfs">DFS</option>
                    <option value="bfs">BFS</option>
                </select>
            )}
        </div>
    );
};

export default Dropdown;