import CytoscapeComponent from 'react-cytoscapejs';
import fcose from 'cytoscape-fcose';
import cytoscape from 'cytoscape';
import { useEffect, useRef, type CSSProperties } from 'react';

cytoscape.use(fcose);

//Stylesheet
const stylesheet=[
  {
    selector: "node",
    style: {
      "label": (ele:any) => {
        return (
          ele.data("__specs-onnx__tensor_node")?.type || ele.data("__specs-onnx__operation_node")?.type || ele.data("__specs-onnx__constant_node")?.value.dataType
        );
      },
      "text-valign": "center",
      color: "#ffffff",
      "font-size": "12px",
      "background-color": "#533b6e",
      /*
      "background-fill": "linear-gradient",
      "background-gradient-stop-colors": "#ff0044 #e5ff00  #00d9ff",
      */
      //"shape": "star",
      "shape": (ele:any) => {
        if(((ele.data("__specs-onnx__tensor_node")?.type || ele.data("__specs-onnx__operation_node")?.type)=="Loop") || ((ele.data("__specs-onnx__tensor_node")?.type || ele.data("__specs-onnx__operation_node")?.type)=="Reshape")){
          return ("circle")
        /*}else if((ele.data("__specs-onnx__tensor_node")?.shape.type || ele.data("__specs-onnx__operation_node")?.shape.type)== "intermediate"){
          return ("vee")*/
        }else if(((ele.data("__specs-onnx__tensor_node")?.type || ele.data("__specs-onnx__operation_node")?.type)=="input") || (ele.data("__specs-onnx__tensor_node")?.type || (ele.data("__specs-onnx__operation_node")?.type)=="output") || ((ele.data("__specs-onnx__tensor_node")?.type || ele.data("__specs-onnx__operation_node")?.type)=="intermediate")){
          return ("pentagon")
        }else{
          return ("diamond")
        }
      },
      width: "40px",
      height: "40px",
    },
  },
  {
    selector: "edge",
    style: {
      width: 2,
      "line-color": "#999",
      "target-arrow-color": "#999",
      "curve-style": "straight",
      "target-arrow-shape": "triangle",
    },
  },
]
//node shape: "star" "ellipse" "circle" "triangle" "pentagon" "tag" "octagon" "vee" "rhomboid" "rectangle" "diamond"
//line style: "straight" "taxi" "segments"

export type CytoscapeData = {
  elements: {
    nodes: Array<Record<string, unknown>>;
    edges: Array<Record<string, unknown>>;
  };
};

export default function CytoscapeGraph(props: {style: CSSProperties, cytoscapeData: CytoscapeData | null, layout: string}) {
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

      cyRef.current.layout({ name: props.layout }).run();
    }, [props.layout]);

    return ( 
      <div style={props.style} ref={containerRef}>
        {props.cytoscapeData && 
        (
          <CytoscapeComponent 
            elements={CytoscapeComponent.normalizeElements(props.cytoscapeData.elements)} 
            style={{ width: "100%", height: "100%" }}
            stylesheet={stylesheet}
            layout={{name: props.layout}}
            cy={(cy) => { cyRef.current = cy; }}
            />
        )
        }
      </div>
    );
}