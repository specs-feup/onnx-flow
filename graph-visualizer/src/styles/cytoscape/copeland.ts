/**
 * @file copeland.ts
 * @description 'Copeland' theme stylesheet for Cytoscape.js graphs.
 * Features radial blue gradient fills, solid light-blue outlines, circle arrow pointers,
 * and high-contrast blue accented edges.
 */

const defaultStylesheet = [

  {
    selector: "node",
    layout: {
      name: "copeland"
    },
    style: {
      "label": (ele:any) => {
        return (
          ele.data("onnxData")?.tensorType || ele.data("onnxData")?.opType || ele.data("onnxData")?.proto.dataType
        );
      },  
      "text-valign": "center",
      color: "#ffffff",
      "font-size": "12px",

      "outline-color": "#8ac7ff",
      "outline-width": "1px",
      "outline-style": "solid",
      "text-outline-color": "rgb(0, 60, 255)",
      "text-outline-width": "1px",

      "background-fill": "radial-gradient",
      "background-gradient-stop-colors": "#63b9db #3077c9 #13389c #13389c ",
      
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
    selector: ':parent',
    style: {
      'background-opacity': 0.25
    }
  },
  {
    selector: "edge",
    style: {
      width: 2,
      "line-color": "#daedff",
      "target-arrow-color": "#daedff",
      "curve-style": "straight",
      "target-arrow-shape": "circle",
    },
  },
  {
    selector: ".cross-graph-capture",
    style: {
      width: 2,
      'line-style': 'dashed',
      "line-color": "#0080ff",
      "target-arrow-color": "#0080ff",
      "curve-style": "straight",
      "target-arrow-shape": "circle",
    },
  },
]
//node shape: "star" "ellipse" "circle" "triangle" "pentagon" "tag" "octagon" "vee" "rhomboid" "rectangle" "diamond"
//line style: "straight" "taxi" "segments" "bezier" "unbundled-bezier" "haystack" "loop"
//pointer style: "triangle" "circle-triangle" "circle" "chevron" "diamond" "tee" "vee" "triangle-tee" "triangle-cross" "triangle-backcurve" "circle" "none"
export default defaultStylesheet;