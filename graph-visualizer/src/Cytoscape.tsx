import CytoscapeComponent from 'react-cytoscapejs';
import fcose from 'cytoscape-fcose';
import cytoscape from 'cytoscape';
import { useEffect, useRef, type CSSProperties } from 'react';
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

export default function CytoscapeGraph(props: {style: CSSProperties, cytoscapeData: CytoscapeData | null, layout: cytoscape.LayoutOptions, nodeColor?: string;}) {
    const cyRef = useRef<cytoscape.Core | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);

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

    return ( 
      <div style={props.style} ref={containerRef}>
        {props.cytoscapeData && 
        (
          <CytoscapeComponent 
            elements={CytoscapeComponent.normalizeElements(props.cytoscapeData.elements)} 
            style={{ width: "100%", height: "100%" }}
            stylesheet={stylesheet}
            layout={props.layout}
            cy={(cy) => { cyRef.current = cy; }}
            />
        )
        }
      </div>
    );
}