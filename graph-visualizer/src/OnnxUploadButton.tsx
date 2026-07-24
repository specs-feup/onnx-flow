import { useState } from 'react';
import type { ChangeEvent, CSSProperties } from 'react';
import type { CytoscapeData } from './Cytoscape.tsx';



export default function OnnxUploadButton(props: {style: CSSProperties, setCytoscapeData: (data: CytoscapeData | null) => void, setFilename: (filename: string | null) => void}) {
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
        props.setCytoscapeData(jsonObject as CytoscapeData);
        props.setFilename(file.name);
      } catch (error: any) {
        // TODO: Validate Cytoscape JSON structure
        if (error.message === 'Invalid Cytoscape JSON') setErrorMessage('Invalid Cytoscape JSON structure. Ensure it has "elements.nodes" and "elements.edges".');
        else setErrorMessage('Failed to parse file. Ensure it is valid JSON syntax.');
        props.setCytoscapeData(null);
        props.setFilename(null);
      }

    };

    // Read the file as raw text
    reader.readAsText(file);

  };

  return (
      <label htmlFor="upload" id="upload-button">Upload your file
        <input id="upload"
          type="file" 
          accept=".json,.onnx" 
          onChange={handleFileUpload}
          hidden 
        />
      </label>
  );

}