'use client';
import React, { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';
import myData from '../../add.json';

const PureCytoscapeGraph = () => {
  // 1. Create a reference to anchor the graph container DOM element
  const containerRef = useRef(null);
  
  // 2. Keep a reference to the core cytoscape instance if you need to mutate it later
  const cyRef = useRef(null);

  useEffect(() => {
    // 3. Initialize Cytoscape when the component mounts
    if (containerRef.current) {
      cyRef.current = cytoscape({
        container: containerRef.current, // Target DOM node
        
        elements: myData.elements,

        style: [
          {
            selector: 'node',
            style: {
              'background-color': '#ff0080',
              'label': 'data(label)',
              'color': '#fff',
              'text-valign': 'center',
              'text-halign': 'center',
            }
          },
          {
            selector: 'edge',
            style: {
              'width': 3,
              'line-color': '#999',
              'target-arrow-color': '#999',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier'
            }
          }
        ],

        layout: {
          name: 'grid',
          rows: 200
        }
      });
    }

    // 4. Clean up the instance when the component unmounts
    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
      }
    };
  }, []); // Empty dependency array runs this strictly on mount

  // 5. Render a styled div container (Width and Height are REQUIRED)
  return (
    <div 
      ref={containerRef} 
      style={{ width: '100%', height: '500px', border: '1px solid #ccc' }} 
    />
  );
};

export default PureCytoscapeGraph;