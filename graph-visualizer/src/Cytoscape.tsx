import CytoscapeComponent from 'react-cytoscapejs';
import fcose from 'cytoscape-fcose';
import cytoscape from 'cytoscape';
import { useEffect, useRef, useState, type CSSProperties } from 'react';
import dagre from 'cytoscape-dagre';
import stylesheet from './styleSheet.ts';

cytoscape.use(dagre);
cytoscape.use(fcose);

export type CytoscapeData = {
  elements: {
    nodes: Array<Record<string, unknown>>;
    edges: Array<Record<string, unknown>>;
  };
};

export default function CytoscapeGraph(props: {style: CSSProperties, cytoscapeData: CytoscapeData | null, layout: cytoscape.LayoutOptions, nodeColor?: string, onNodeSelected?: (node:any, pos:{x:number,y:number})=>void}) {
  const cyRef = useRef<cytoscape.Core | null>(null);
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
      if (!props.nodeColor) return;
      cyRef.current.nodes().style('background-color', props.nodeColor);
      cyRef.current.resize();
      cyRef.current.fit();

    }, [props.nodeColor, props.cytoscapeData]);

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