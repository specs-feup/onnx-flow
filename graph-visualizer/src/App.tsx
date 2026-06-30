import { useState } from 'react'

import OnnxUploadButton from './OnnxUploadButton.tsx'
//import GraphVisualizer from './GraphVisualizer.tsx'
import './App.css'

function App() {
  const [count, setCount] = useState(0)
  return (
    <main id="spacer">
      <OnnxUploadButton />
    </main>
  )
}

export default App
