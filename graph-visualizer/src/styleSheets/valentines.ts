const valentines = [
    {
        selector: "node",
        layout: {
            name: "valentines",
        },
        style: {
            label: (ele: any) => {
                return (
                    ele.data("onnxData")?.tensorType ||
                    ele.data("onnxData")?.opType ||
                    ele.data("onnxData")?.proto.dataType
                );
            },
            "text-valign": "center",
            color: "#ffffff",
            "font-size": "12px",
            "background-color": "#e94993",

            "outline-color": "#d3234f",
            "outline-width": "2px",
            "outline-style": "solid",
            //"shape": "star",
            shape: (ele: any) => {
                if (ele.data("onnxData")?.kind == "OperationNode") {
                    return "circle";
                } else if (ele.data("onnxData")?.kind == "TensorNode") {
                    return "pentagon";
                } else {
                    return "diamond";
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
            "line-color": "rgb(255, 159, 215)",
            "target-arrow-color": "rgb(255, 159, 215)",
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
            "line-style": "dashed",
            "line-color": "#83001c",
            "target-arrow-color": "#83001c",
            "curve-style": "straight",
            "target-arrow-shape": "vee",
        },
    },
];
//node shape: "star" "ellipse" "circle" "triangle" "pentagon" "tag" "octagon" "vee" "rhomboid" "rectangle" "diamond"
//line style: "straight" "taxi" "segments" "bezier" "unbundled-bezier" "haystack" "loop"
//pointer style: "triangle" "circle-triangle" "circle" "chevron" "diamond" "tee" "vee" "triangle-tee" "triangle-cross" "triangle-backcurve" "circle" "none"
export default valentines;
