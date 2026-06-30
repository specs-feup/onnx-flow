import { useState } from 'react';
import type { ChangeEvent } from 'react';
import ReactDOM from 'react-dom';
import CytoscapeComponent from 'react-cytoscapejs';
import fcose from 'cytoscape-fcose';
import cytoscape from 'cytoscape';
import React from 'react';

cytoscape.use(fcose);

type CytoscapeData = {
  elements: {
    nodes: Array<Record<string, unknown>>;
    edges: Array<Record<string, unknown>>;
  };
};

export default function OnnxUploadButton(){
  const [parsedData, setParsedData] = useState<CytoscapeData | null>(null);
  const [errorMessage, setErrorMessage] = useState('');

  const handleFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
    // Grab the first file selected by the user
    const file = event.target.files?.[0];
    if (!file) return;

    // Clear any old error messages
    setErrorMessage('');

    // Create a new FileReader instance
    const reader = new FileReader();

    // Define what happens when the file is done reading
    reader.onload = (e) => {
      try {
        const textContent = e.target?.result as string;
        const jsonObject = JSON.parse(textContent) as Partial<CytoscapeData>;
        if (!jsonObject.elements || !jsonObject.elements.nodes || !jsonObject.elements.edges) {
          throw new Error('Invalid Cytoscape JSON');
        }
        setParsedData(jsonObject as CytoscapeData);
      } catch (error: any) {
        // TODO: Validate Cytoscape JSON structure
        if (error.message === 'Invalid Cytoscape JSON') setErrorMessage('Invalid Cytoscape JSON structure. Ensure it has "elements.nodes" and "elements.edges".');
        else setErrorMessage('Failed to parse file. Ensure it is valid JSON syntax.');
        setParsedData(null);
      }

    };

    // Read the file as raw text
    reader.readAsText(file);

  };

  return (
    <div style={{ padding: '20px' }}>
      <h3>Upload and Parse JSON File</h3>
      
      {/* Accept attribute safely filters for only .json files */}
      
      <input id="upload"
        type="file" 
        accept=".json" 
        onChange={handleFileUpload}
        hidden 
      />
      <label htmlFor="upload">Upload your file</label>

      {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}

      {parsedData && 
      <div style={{width: '60vw', height: '80vh',marginTop: '23px', border: '2px solid white'}}>
        <CytoscapeComponent elements={CytoscapeComponent.normalizeElements(parsedData.elements)} style={{width: '100%', height: '100%' }} layout={{name: 'breadthfirst'}} />
      </div>
      }
    </div>
  );

}