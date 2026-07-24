
const stylesheet=[
  {
    selector: "node",
    style: {
      "label": (ele:any) => {
        return (
          ele.data("onnxData")?.tensorType || ele.data("onnxData")?.opType || ele.data("onnxData")?.proto.dataType
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
        if(((ele.data("onnxData")?.tensorType || ele.data("onnxData")?.opType)=="Loop") || ((ele.data("onnxData")?.tensorType || ele.data("onnxData")?.opType)=="Reshape")){
          return ("circle")
        }else if (((ele.data("onnxData")?.tensorType || ele.data("onnxData")?.opType)=="input") || ((ele.data("onnxData")?.tensorType || ele.data("onnxData")?.opType)=="output") || ((ele.data("onnxData")?.tensorType || ele.data("onnxData")?.opType)=="intermediate")){
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
//line style: "straight" "taxi" "segments" "bezier" "unbundled-bezier" "haystack" "loop"
//pointer style: "triangle" "circle-triangle" "circle" "chevron" "diamond" "tee" "vee" "triangle-tee" "triangle-cross" "triangle-backcurve" "circle" "none"
export default stylesheet;