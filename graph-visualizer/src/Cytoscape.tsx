import CytoscapeComponent from 'react-cytoscapejs';
import fcose from 'cytoscape-fcose';
import cytoscape from 'cytoscape';
import { useEffect, useRef, useState, type CSSProperties } from 'react';
import dagre from 'cytoscape-dagre';
import cxtmenu from 'cytoscape-cxtmenu';
import stylesheet from './styleSheets/styleSheet.ts';
import chroma from 'chroma-js';

cytoscape.use(dagre);
cytoscape.use(fcose);

cytoscape.use(cxtmenu);

export type CytoscapeData = {
  elements: {
    nodes: Array<Record<string, unknown>>;
    edges: Array<Record<string, unknown>>;
  };
};

export default function CytoscapeGraph(props: {style: CSSProperties, cytoscapeData: CytoscapeData | null, layout: cytoscape.LayoutOptions, nodeColor?: string, selectedNodeId?: string | null, onNodeSelected?: (node:any, pos:{x:number,y:number})=>void}) {
  const cyRef = useRef<cytoscape.Core | null>(null);
  const menuRef = useRef(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [cyReady, setCyReady] = useState(false);

    useEffect(() => {
        if (!containerRef.current) return;

        const resizeObserver = new ResizeObserver(() => {
            if (cyRef.current) {
                cyRef.current.resize();
                cyRef.current.fit();
            }
        });

        resizeObserver.observe(containerRef.current);

        return () => {resizeObserver.disconnect();};
        }, []);
        
    useEffect(() => {
      if (!cyRef.current) return;

      cyRef.current.layout(props.layout ).run();
    }, [props.layout]);

    useEffect(() => {
      if (!cyRef.current) return;
      const defaultColor = props.nodeColor || '#533b6e';
      cyRef.current.nodes().style('background-color', defaultColor);
      if (props.selectedNodeId) {
        const selected = cyRef.current.getElementById(props.selectedNodeId);
        if (selected && selected.nonempty()) {
          selected.style('background-color', chroma(defaultColor).brighten(2).hex() ); 
        }
      }
      cyRef.current.resize();
    }, [props.nodeColor, props.selectedNodeId, props.cytoscapeData]);

    useEffect(() => {
      if (!cyReady || !cyRef.current || !props.onNodeSelected) return;

      const handler = (event: any) => {
      if (!event.target?.isNode?.()) return;
      
      const pos = event.renderedPosition;
      const rect = containerRef.current?.getBoundingClientRect();
      const screenPos = rect 
        ? { x: rect.left + pos.x, y: rect.top + pos.y }
        : { x: 10, y: 10 };
      
      props.onNodeSelected(event.target.data(), screenPos);
      };

      cyRef.current.on('tap', 'node', handler);
      return () => cyRef.current?.off('tap', 'node', handler);
    }, [cyReady, props.onNodeSelected]);

    useEffect(() => {
      if (!cyReady || !cyRef.current) return;
      
      const cy = cyRef.current;
      if (menuRef.current) {
        menuRef.current.destroy();
      }
      
      menuRef.current = cy.cxtmenu({
      selector: 'node', // Options: 'node', 'edge', or 'core' (for background)
      activeFillColor: '#533b6e8e',
      commands: [
        {
          fillColor:  'rgba(64, 67, 75, 0.9)',
          content: 'Log Info',
          select: function(ele) {
            console.log('Selected node ID:', ele.id());
          }
        },
        {
          fillColor: 'rgba(75, 26, 38, 0.9)',

          content: 'Delete',
          select: function(ele) {
            cy.remove(ele); // Manipulate the graph directly via the API
          }
        }
      ]
    });

    // Variable to temporarily store the click coordinate location
    let clickedPosition = { x: 0, y: 0 };

    cy.on('cxttapstart', (event) => {
      // event.target is the core graph. target.pointer contains the exact canvas model coordinates
      if (event.target && event.target.pointer) {
        console.log('Context menu opened at model coordinates:', event.target.pointer);
        clickedPosition = {
          x: event.target.pointer.x,
          y: event.target.pointer.y
        };
      }
    });

    // Set up the context menu for the background canvas ('core')
    menuRef.current = cy.cxtmenu({
      selector: 'core', // Targets the empty background space
      activeFillColor: '#533b6e8e',
      commands: [
        {
          fillColor: 'rgba(32, 70, 92, 0.79)', // Green background for creation
          content: '＋ Add Node',
          select: function() {
            const newId = `node`;
            
            // Add the new node directly into cytoscape at the recorded position
            cy.add({
              group: 'nodes',
              data: { 
                id: newId, 
                label: `New Node (${newId})` 
              },
              // Use the model position captured when the menu opened
              position: { 
                x: clickedPosition.x, 
                y: clickedPosition.y 
              }
            });
          }
        }
      ]
    });

    return () => {
      if (menuRef.current) {
        menuRef.current.destroy();
        menuRef.current = null;
      }
    }

    }, [cyReady])

    return ( 
      <div style={props.style} ref={containerRef}>
        {props.cytoscapeData && 
        (
          <CytoscapeComponent 
            elements={CytoscapeComponent.normalizeElements(props.cytoscapeData.elements)} 
            style={{ width: "100%", height: "100%" }}
            stylesheet={stylesheet}
            layout={props.layout}
            cy={(cy) => { cyRef.current = cy; setCyReady(true); }}
            />
        )
        }
      </div>
    );
}