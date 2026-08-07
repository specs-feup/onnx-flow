
const defaultStylesheet=[
  {
    selector: "node",
    layout: {
      name: "nge"
    },
    style: {
      "label": (ele:any) => {
        return (
          ele.data("onnxData")?.tensorType || ele.data("onnxData")?.opType || ele.data("onnxData")?.proto.dataType
        );
      },
      "text-valign": "center",
      color: "#e07f00",
      "font-size": "12px",
      "background-color": "#bd0000",
      "text-border-width": 2,
      "text-outline-width": 2,
      "text-outline-color": "rgb(0, 0, 0)",
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
    selector: ':parent',
    style: {
      'background-opacity': 0.25
    }
  },
  {
    selector: "edge",
    style: {
      width: 2,
      "line-color": "#000000",
      "line-outline-color": "#bd0000",
      "line-outline-width": 1,
      "target-arrow-color": "#bd0000",
      "curve-style": "straight",
      "target-arrow-outline-color": "#bd0000",
      "target-arrow-outline-width": 1,
      "target-arrow-shape": "vee",
    },
  },
  {
    selector: ".cross-graph-capture",
    style: {
      width: 2,
      'line-style': 'dashed',
      "line-color": "#ca6225",
      "line-outline-color": "#222222",
      "line-outline-width": 1,
      "target-arrow-color": "#ca6225",
      "curve-style": "straight",
      "target-arrow-shape": "vee",
    },
  },
]
//node shape: "star" "ellipse" "circle" "triangle" "pentagon" "tag" "octagon" "vee" "rhomboid" "rectangle" "diamond"
//line style: "straight" "taxi" "segments" "bezier" "unbundled-bezier" "haystack" "loop"
//pointer style: "triangle" "circle-triangle" "circle" "chevron" "diamond" "tee" "vee" "triangle-tee" "triangle-cross" "triangle-backcurve" "circle" "none"
export default defaultStylesheet;