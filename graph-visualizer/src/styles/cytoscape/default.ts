/**
 * @file default.ts
 * @description Default visual theme stylesheet for Cytoscape.js graphs.
 * Distinguishes OperationNodes (circles), TensorNodes (pentagons), and ConstantNodes (diamonds)
 * with purple node styling, directional arrow edges, and dashed cross-graph capture edges.
 */

const defaultStylesheet = [

  {
    selector: "node",
    layout: {
      name: "default"
    },
    style: {
      "label": (ele:any) => {
        return (
          ele.data("onnxData")?.tensorType || ele.data("onnxData")?.opType || ele.data("onnxData")?.proto?.dataType
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
        if((ele.data("onnxData")?.kind == "OperationNode")){
          return ("circle")
        }else if ((ele.data("onnxData")?.kind == "TensorNode")){
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
  {
    selector: ':parent',
    style: {
      'background-opacity': 0.25
    }
  },/*
  {
    selector: 'node[parent]',
    style: {
      'width': '10px',
      'height': '10px',
      'font-size': '12px'
    }
  },
  {
    selector: 'edge[parent]',
    style: {
      'width': '4px',
      'font-size': '10px'
    }
  },*/
  {
    selector: ".cross-graph-capture",
    style: {
      width: 2,
      'line-style': 'dashed',
      "line-color": "#b4bfca",
      "target-arrow-color": "#b4bfca",
      "curve-style": "straight",
      "target-arrow-shape": "triangle",
    },
  },
]
//node shape: "star" "ellipse" "circle" "triangle" "pentagon" "tag" "octagon" "vee" "rhomboid" "rectangle" "diamond"
//line style: "straight" "taxi" "segments" "bezier" "unbundled-bezier" "haystack" "loop"
//pointer style: "triangle" "circle-triangle" "circle" "chevron" "diamond" "tee" "vee" "triangle-tee" "triangle-cross" "triangle-backcurve" "circle" "none"
export default defaultStylesheet;