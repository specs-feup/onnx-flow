
const defaultStylesheet=[
  {
    selector: "node",
    layout: {
      name: "default"
    },
    style: {
      "label": (ele:any) => {
        return (
          ele.data("onnxData")?.tensorType || ele.data("onnxData")?.opType || ele.data("onnxData")?.proto.dataType
        );
      },
      "text-valign": "center",
      color: "#000000",
      "font-size": "12px",
      "background-color": "#bd0000",

      "outline-color": "#e07f00",
      "outline-width": "2px",
      "outline-style": "solid",
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
      "line-color": "#000000",
      "target-arrow-color": "#000000",
      "curve-style": "straight",
      "target-arrow-shape": "triangle",
    },
  },
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