import { useState } from 'react';
import type { ChangeEvent, CSSProperties } from 'react';
import type { CytoscapeData } from './Cytoscape.tsx';



export default function OnnxUploadButton(props: {style: CSSProperties, setCytoscapeData: (data: CytoscapeData | null) => void}) {
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
        props.setCytoscapeData(jsonObject as CytoscapeData);
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
      <label htmlFor="upload" id="upload-button">Upload your file
        <input id="upload"
          type="file" 
          accept=".json" 
          onChange={handleFileUpload}
          hidden 
        />
      </label>
  );

}