/**
 * @file mono.ts
 * @description 'Mono' theme stylesheet for Cytoscape.js graphs.
 * Features a minimalist monochromatic black and white aesthetic with chevron arrow pointers.
 */

const mono = [

  {
    selector: "node",
    layout: {
      name: "mono"
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
      "background-color": "#000000",
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
    selector: ':parent',
    style: {
      'background-opacity': 0.25
    }
  },
  {
    selector: "edge",
    style: {
      width: 2,
      "line-color": "#ffffff",
      "target-arrow-color": "#ffffff",
      "curve-style": "straight",
      "target-arrow-shape": "chevron",
    },
  },
  {
    selector: ".cross-graph-capture",
    style: {
      width: 2,
      'line-style': 'dashed',
      "line-color": "#808080",
      "target-arrow-color": "#808080",
      "curve-style": "straight",
      "target-arrow-shape": "chevron",
    },
  },
]
//node shape: "star" "ellipse" "circle" "triangle" "pentagon" "tag" "octagon" "vee" "rhomboid" "rectangle" "diamond"
//line style: "straight" "taxi" "segments" "bezier" "unbundled-bezier" "haystack" "loop"
//pointer style: "triangle" "circle-triangle" "circle" "chevron" "diamond" "tee" "vee" "triangle-tee" "triangle-cross" "triangle-backcurve" "circle" "none"
export default mono;