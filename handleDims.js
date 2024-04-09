function getAddDimensions(edges, index, inputEdgesWithKnownDims, outputEdgesWithKnownDims, inputEdgesWithUnknownDims, outputEdgesWithUnknownDims) {
    let otherInputIndex = inputEdgesWithKnownDims.findIndex((e, idx) => idx !== index && e.data.target === inputEdgesWithKnownDims[index].data.target);
    let outputIndex =  outputEdgesWithKnownDims.findIndex(e => e.data.source === inputEdgesWithKnownDims[index].data.target);
    if (otherInputIndex !== -1 && outputIndex !== -1) {
        const inputToOtherNodeIndex = inputEdgesWithUnknownDims.findIndex(e => e.data.source === outputEdgesWithKnownDims[outputIndex].data.target)
        

        edges.push(inputEdgesWithKnownDims[index]);
        edges.push(inputEdgesWithKnownDims[otherInputIndex]);
        edges.push(outputEdgesWithKnownDims[outputIndex]);
    
    
        inputEdgesWithKnownDims.splice(index, 1)[0]
        inputEdgesWithKnownDims.splice(otherInputIndex - 1, 1)[0]
        outputEdgesWithKnownDims.splice(outputIndex, 1)[0]
    }
    
    else if (otherInputIndex !== -1 && outputIndex === -1) {
        outputIndex = outputEdgesWithUnknownDims.findIndex(e => e.data.source === inputEdgesWithKnownDims[index].data.target);
        if (outputIndex !== -1) {
            outputEdgesWithUnknownDims[outputIndex].dimensions = inputEdgesWithKnownDims[index].dimensions;
            outputEdgesWithUnknownDims[outputIndex].elemType = inputEdgesWithKnownDims[index].elemType;

            const inputToOtherNodeIndex = inputEdgesWithUnknownDims.findIndex(e => e.data.source === outputEdgesWithUnknownDims[outputIndex].data.target)
            if (inputToOtherNodeIndex !== -1) {
                inputEdgesWithUnknownDims[inputToOtherNodeIndex].dimensions = inputEdgesWithKnownDims[index].dimensions
                inputEdgesWithUnknownDims[inputToOtherNodeIndex].elemType = inputEdgesWithKnownDims[index].elemType
                inputEdgesWithKnownDims.push(inputEdgesWithUnknownDims.splice(inputToOtherNodeIndex, 1)[0])
            }
            
            edges.push(inputEdgesWithKnownDims[index]);
            edges.push(inputEdgesWithKnownDims[otherInputIndex]);
            edges.push(outputEdgesWithUnknownDims[outputIndex]);
        
        
            inputEdgesWithKnownDims.splice(index, 1)[0]
            inputEdgesWithKnownDims.splice(otherInputIndex - 1, 1)[0]
            outputEdgesWithUnknownDims.splice(outputIndex, 1)[0]
        }
    }
    else if (otherInputIndex === -1 && outputIndex !== -1) {

        otherInputIndex = inputEdgesWithUnknownDims.findIndex(e => e.data.target === inputEdgesWithKnownDims[index].data.target);
        if (otherInputIndex !== -1) {
            inputEdgesWithUnknownDims[otherInputIndex].dimensions = inputEdgesWithKnownDims[index].dimensions;
            inputEdgesWithUnknownDims[otherInputIndex].elemType = inputEdgesWithKnownDims[index].elemType;

            const outputToOtherNodeIndex = outputEdgesWithUnknownDims.findIndex(e => e.data.target === inputEdgesWithUnknownDims[otherInputIndex].data.source)
            if (outputToOtherNodeIndex !== -1) {
                outputEdgesWithUnknownDims[outputToOtherNodeIndex].dimensions = inputEdgesWithKnownDims[index].dimensions
                outputEdgesWithUnknownDims[outputToOtherNodeIndex].elemType = inputEdgesWithKnownDims[index].elemType
                outputEdgesWithKnownDims.push(outputEdgesWithUnknownDims.splice(outputToOtherNodeIndex, 1)[0])
            }
            
            edges.push(inputEdgesWithKnownDims[index]);
            edges.push(inputEdgesWithUnknownDims[otherInputIndex]);
            edges.push(outputEdgesWithKnownDims[outputIndex]);
        
            inputEdgesWithKnownDims.splice(index, 1)[0]
            inputEdgesWithUnknownDims.splice(otherInputIndex, 1)[0]
            outputEdgesWithKnownDims.splice(outputIndex, 1)[0]
        }
    }
    
    else {
        outputIndex = outputEdgesWithUnknownDims.findIndex(e => e.data.source === inputEdgesWithKnownDims[otherInputIndex].data.target);
        otherInputIndex = inputEdgesWithUnknownDims.findIndex(e => e.data.target === inputEdgesWithKnownDims[index].data.target);
        if (outputIndex !== -1 && otherInputIndex !== -1) {
            inputEdgesWithUnknownDims[otherInputIndex].dimensions = inputEdgesWithKnownDims[index].dimensions;
            inputEdgesWithUnknownDims[otherInputIndex].elemType = inputEdgesWithKnownDims[index].elemType;
            outputEdgesWithUnknownDims[outputIndex].dimensions = inputEdgesWithKnownDims[index].dimensions;
            outputEdgesWithUnknownDims[outputIndex].elemType = inputEdgesWithKnownDims[index].elemType;

            const inputToOtherNodeIndex = inputEdgesWithUnknownDims.findIndex(e => e.data.source === outputEdgesWithUnknownDims[outputIndex].data.target)
            if (inputToOtherNodeIndex !== -1) {
                inputEdgesWithUnknownDims[inputToOtherNodeIndex].dimensions = inputEdgesWithKnownDims[index].dimensions
                inputEdgesWithUnknownDims[inputToOtherNodeIndex].elemType = inputEdgesWithKnownDims[index].elemType
                inputEdgesWithKnownDims.push(inputEdgesWithUnknownDims.splice(inputToOtherNodeIndex, 1)[0])
            }

            const outputToOtherNodeIndex = outputEdgesWithUnknownDims.findIndex(e => e.data.target === inputEdgesWithUnknownDims[otherInputIndex].data.source)
            if (outputToOtherNodeIndex !== -1) {
                outputEdgesWithUnknownDims[outputToOtherNodeIndex].dimensions = inputEdgesWithKnownDims[index].dimensions
                outputEdgesWithUnknownDims[outputToOtherNodeIndex].elemType = inputEdgesWithKnownDims[index].elemType
                outputEdgesWithKnownDims.push(outputEdgesWithUnknownDims.splice(outputToOtherNodeIndex, 1)[0])
            }

            edges.push(inputEdgesWithKnownDims[index]);
            edges.push(inputEdgesWithUnknownDims[otherInputIndex]);
            edges.push(outputEdgesWithUnknownDims[outputIndex]);
        
            inputEdgesWithKnownDims.splice(index, 1)[0]
            inputEdgesWithUnknownDims.splice(otherInputIndex, 1)[0]
            outputEdgesWithUnknownDims.splice(outputIndex, 1)[0]
        }
    }
    
}


function getMatMulDimensions(edges, index, inputEdgesWithKnownDims, outputEdgesWithKnownDims, inputEdgesWithUnknownDims, outputEdgesWithUnknownDims) {
    let otherInputIndex = inputEdgesWithKnownDims.findIndex((e, idx) => idx !== index && e.data.target === inputEdgesWithKnownDims[index].data.target);
    let outputIndex =  outputEdgesWithKnownDims.findIndex(e => e.data.source === inputEdgesWithKnownDims[index].data.target);

    if (otherInputIndex !== -1 && outputIndex !== -1) { 

        edges.push(inputEdgesWithKnownDims[index]);
        edges.push(inputEdgesWithKnownDims[otherInputIndex]);
        edges.push(outputEdgesWithKnownDims[outputIndex]);
    
    
        inputEdgesWithKnownDims.splice(index, 1)[0]
        inputEdgesWithKnownDims.splice(otherInputIndex - 1, 1)[0]
        outputEdgesWithKnownDims.splice(outputIndex, 1)[0]
        return 0
    }
    if (otherInputIndex !== -1 && outputIndex === -1) {
        
        outputIndex = outputEdgesWithUnknownDims.findIndex(e => e.data.source === inputEdgesWithKnownDims[index].data.target);
        if (outputIndex !== -1) {
            outputEdgesWithUnknownDims[outputIndex].elemType = inputEdgesWithKnownDims[index].elemType;
            if (inputEdgesWithKnownDims[index].dimensions[1].dimValue === inputEdgesWithKnownDims[otherInputIndex].dimensions[0].dimValue) {
                outputEdgesWithUnknownDims[outputIndex].dimensions[0] = inputEdgesWithKnownDims[index].dimensions[0];
                outputEdgesWithUnknownDims[outputIndex].dimensions[1] = inputEdgesWithKnownDims[otherInputIndex].dimensions[1];
            }
            else {
                outputEdgesWithUnknownDims[outputIndex].dimensions[0] = inputEdgesWithKnownDims[index].dimensions[0];
                outputEdgesWithUnknownDims[outputIndex].dimensions[1] = inputEdgesWithKnownDims[otherInputIndex].dimensions[1];
            }


            const inputToOtherNodeIndex = inputEdgesWithUnknownDims.findIndex(e => e.data.source === outputEdgesWithUnknownDims[outputIndex].data.target)
            if (inputToOtherNodeIndex !== -1) {
                inputEdgesWithUnknownDims[inputToOtherNodeIndex].dimensions = outputEdgesWithUnknownDims[outputIndex].dimensions
                inputEdgesWithUnknownDims[inputToOtherNodeIndex].elemType = outputEdgesWithUnknownDims[outputIndex].elemType
                inputEdgesWithKnownDims.push(inputEdgesWithUnknownDims.splice(inputToOtherNodeIndex, 1)[0])
            }
            
            edges.push(inputEdgesWithKnownDims[index]);
            edges.push(inputEdgesWithKnownDims[otherInputIndex]);
            edges.push(outputEdgesWithUnknownDims[outputIndex]);
        
        
            inputEdgesWithKnownDims.splice(index, 1)[0]
            inputEdgesWithKnownDims.splice(otherInputIndex - 1, 1)[0]
            outputEdgesWithUnknownDims.splice(outputIndex, 1)[0]
        
            return 0 
        }
    }
    
    if (otherInputIndex === -1 && outputIndex !== -1) {
        otherInputIndex = inputEdgesWithUnknownDims.findIndex(e => e.data.target === inputEdgesWithKnownDims[index].data.target);
        if (otherInputIndex !== -1) {
            inputEdgesWithUnknownDims[otherInputIndex].elemType = inputEdgesWithKnownDims[index].elemType;
            if (inputEdgesWithKnownDims[index].dimensions[0].dimValue === outputEdgesWithKnownDims[outputIndex].dimensions[0].dimValue) {
                inputEdgesWithUnknownDims[otherInputIndex].dimensions[0] = inputEdgesWithKnownDims[index].dimensions[1];
                inputEdgesWithUnknownDims[otherInputIndex].dimensions[1] = outputEdgesWithKnownDims[outputIndex].dimensions[1];
            }
            else {
                inputEdgesWithUnknownDims[otherInputIndex].dimensions[1] = inputEdgesWithKnownDims[index].dimensions[0];
                inputEdgesWithUnknownDims[otherInputIndex].dimensions[0] = outputEdgesWithKnownDims[outputIndex].dimensions[0];
            }
            const outputToOtherNodeIndex = outputEdgesWithUnknownDims.findIndex(e => e.data.target === inputEdgesWithUnknownDims[otherInputIndex].data.source)
            if (outputToOtherNodeIndex !== -1) {
                outputEdgesWithUnknownDims[outputToOtherNodeIndex].dimensions = inputEdgesWithKnownDims[index].dimensions
                outputEdgesWithUnknownDims[outputToOtherNodeIndex].elemType = inputEdgesWithKnownDims[index].elemType
                outputEdgesWithKnownDims.push(outputEdgesWithUnknownDims.splice(outputToOtherNodeIndex, 1)[0])
            }
            
            edges.push(inputEdgesWithKnownDims[index]);
            edges.push(inputEdgesWithUnknownDims[otherInputIndex]);
            edges.push(outputEdgesWithKnownDims[outputIndex]);
        
            inputEdgesWithKnownDims.splice(index, 1)[0]
            inputEdgesWithUnknownDims.splice(otherInputIndex, 1)[0]
            outputEdgesWithKnownDims.splice(outputIndex, 1)[0]

            return 0
        }
    }
    return 1
}


export function getAllDimensions(inputEdgesWithKnownDims, outputEdgesWithKnownDims, inputEdgesWithUnknownDims, outputEdgesWithUnknownDims) {
    let edges = []

    while (inputEdgesWithKnownDims.length) {
        let index = 0;
        while (index < inputEdgesWithKnownDims.length) {
        
            if (inputEdgesWithKnownDims[index].opType === 'Add') {
                getAddDimensions(edges, index, inputEdgesWithKnownDims, outputEdgesWithKnownDims, inputEdgesWithUnknownDims, outputEdgesWithUnknownDims)
            }
            
            else if (inputEdgesWithKnownDims[index].opType === 'MatMul') {
                index = getMatMulDimensions(edges, index, inputEdgesWithKnownDims, outputEdgesWithKnownDims, inputEdgesWithUnknownDims, outputEdgesWithUnknownDims)
            }
        }
    }
    return edges
}