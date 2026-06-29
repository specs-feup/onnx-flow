'use client';

import React from 'react';
import CytoscapeComponent from 'react-cytoscapejs';

export default function MyGraphApp() {

    const elements = [
    { data: { id: 'one', label: 'Node 1' } },
    { data: { id: 'two', label: 'Node 2' } },
    { data: { id: 'edge1', source: 'one', target: 'two' } }
    ];

    // Visual styling rules for your graph
    const stylesheet = [
    {
        selector: 'node',
        style: {
        'background-color': '#0074D9',
        'label': 'data(label)', // Pulls the label text dynamically from the node data
        'color': '#333',
        'font-size': '12px'
        }
    },
    {
        selector: 'edge',
        style: {
        'width': 3,
        'line-color': '#999',
        'target-arrow-color': '#999',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier' // Essential for arrows and curves to render properly
        }
    }
    ];

// Layout dictates how nodes are positioned on the initial render
    const layout = { name: 'grid', rows: 1 };
  return (
    <div style={{ padding: '20px' }}>
      <h1>My Cytoscape Network</h1>
      
      <div style={{ border: '1px solid #ccc', width: '500px', height: '500px' }}>
        <CytoscapeComponent
          elements={elements}
          stylesheet={stylesheet}
          layout={layout}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
}