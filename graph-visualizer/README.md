# ONNX Graph Visualizer & Editor

A modern, interactive web application built with **React 19**, **TypeScript**, **Vite**, and **Cytoscape.js** for visualizing, inspecting, transforming, and editing ONNX (Open Neural Network Exchange) computation graphs.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
  - [1. Homepage & File Explorer](#1-homepage--file-explorer)
  - [2. Visualizer Mode](#2-visualizer-mode)
  - [3. Editor Mode](#3-editor-mode)
  - [4. Model Transformation & History](#4-model-transformation--history)
  - [5. Export & Compilation](#5-export--compilation)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Project Structure](#project-structure)
- [Backend API Integration](#backend-api-integration)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)

---

## 🔍 Overview

The **ONNX Graph Visualizer & Editor** acts as a visual interface for managing ONNX models processed by `@specs-feup/onnx-flow`. It allows developers and researchers to explore graph structures, inspect tensor shapes and operator schemas, apply graph transformation recipes (such as node fusions and rewrites), and interactively construct or edit ONNX graphs in real time.

---

## ✨ Key Features

### 1. Homepage & File Explorer (`/`)
- **Available ONNX Files List**: Fetches and displays available ONNX files on the backend server.
- **Search & Filtering**: Live search bar to filter model files by name.
- **Sorting Options**: Sort files by Name, Size (in bytes), or Last Modified date in Ascending or Descending order.
- **Session Launching**: Select a file to launch a new graph visualization session in a dedicated workspace tab (`POST /api/sessions`).

### 2. Visualizer Mode (`/app/:sessionId`)
- **Interactive Graph Canvas**: Powered by Cytoscape.js with full pan, zoom, drag, node grouping, and selection capabilities.
- **Multiple Layout Algorithms**:
  - **fcose** (Fast Compound Spring Embedder - Default)
  - **Dagre** (Hierarchical / Directed: Left-to-Right `LR` & Top-to-Bottom `TB`)
  - **ELK** (Eclipse Layout Kernel: Layered Left-to-Right `RIGHT` & Top-to-Bottom `DOWN`)
  - **BFS** (Breadth-First Search layout)
  - **Grid** & **Concentric (Circle)** layouts
- **Theme & Aesthetic Customization**:
  - Prebuilt visual themes (`default`, `copeland`, `nerv`/NGE, `mono`, `eyestrain`, `valentines`, `easteregg`).
  - Interactive **Color Picker** for node highlights and custom colors.
- **Control Flow & Compound Region Visualization**:
  - Renders subgraphs (e.g. `If` branches, `Loop` bodies, `Scan` regions) as nested Cytoscape compound nodes.
  - Interactive radial context menus to expand/collapse individual region branches (`Expand then_branch`, `Collapse body`) or collapse/expand all subgraphs at once.
  - **Loop Viewer Window**: Modal interface providing an isolated view of nested graph regions inside operation nodes.
- **Inspection Popups**:
  - **Node Details Window**: Inspects node types (`TensorNode`, `OperationNode`, `ConstantNode`), data types, shapes, shape metadata, operation attributes, and input connections.
  - **Edge Details Window**: Displays source, target, edge ID, and inner target details.

### 3. Editor Mode (`/app/:sessionId`)
- **Interactive Graph Construction & Editing**: Toggle to Editor mode to modify existing nodes or add new nodes.
- **Adding Nodes**:
  - Add nodes directly via right-click canvas context menu (`＋ Add Node`).
  - Generate random Node IDs or specify custom identifiers.
  - **Constant Node**: Define constant name, data type (`FLOAT`, `DOUBLE`, `INT32`, `INT64`, `STRING`, `BOOL`, `FLOAT16`, etc.), multi-dimensional shape, and raw tensor data values.
  - **Tensor Node**: Specify literal data type, tensor category (`Input`, `Output`, `Intermediate`, `Index`, `Index_Aux`), and dynamic shape dimensions (`DimensionBuilder`).
  - **Operation Node**: Select from standard ONNX operation schemas (`StandardOps`), bind input slots to existing value nodes (with support for variadic inputs and optional flags), edit schema-validated attributes, and automatically generate schema output nodes.
- **Editing & Deleting Nodes**:
  - Right-click context menu "Edit" action opens pre-populated node configuration fields.
  - Right-click context menu "Delete" action performs cascading node deletion, removing descendant region nodes and attached edges.
- **Session Persistence & Recovery**:
  - Unsaved graph modifications are automatically cached in browser `sessionStorage`.
  - **Restore Modal**: Prompts the user upon page re-entry to restore previous unsaved graph edits or discard changes.

### 4. Model Transformation & History
- **Opportunity Discovery**: Queries backend for available graph transformation recipes (e.g., node fusions, canonicalization rules).
- **One-Click Transformations**: Apply transformation recipes to target nodes directly from the side panel.
- **Undo / Redo System**: Full transformation step history tracking with ↩ Undo and ↪ Redo functionality.

### 5. Export & Compilation
- **ONNX Model Compilation**: Compiles edited in-memory graphs back into ONNX representations via backend compiler (`POST /api/sessions/:sessionId/compile`).
- **Compile Feedback Modal**: Visual confirmation of compilation success or detailed error stack trace reports.
- **Model Export**: Download graphs in two formats:
  - **ONNX in JSON** (`/api/sessions/:sessionId/export/onnx-json`)
  - **Unified JSON** (`/api/sessions/:sessionId/export/unified-json`)

---

## 🛠 Architecture & Tech Stack

- **Framework**: [React 19](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- **Build Tool**: [Vite 8](https://vite.dev/)
- **Graph Engine**: [Cytoscape.js](https://js.cytoscape.org/) + Layout Extensions (`cytoscape-fcose`, `cytoscape-dagre`, `cytoscape-elk`, `cytoscape-cxtmenu`, `cytoscape-expand-collapse`)
- **UI Components & Styling**:
  - [Radix UI Dropdown Menu](https://www.radix-ui.com/)
  - [React Select](https://react-select.com/)
  - [Mantine Core](https://mantine.dev/)
  - [Bootstrap 5](https://getbootstrap.com/)
  - Custom CSS Design Tokens & Themes
- **Backend API Communication**: Express REST API calls (`fetch` + Beacon API)

---

## 📁 Project Structure

```
graph-visualizer/
├── src/
│   ├── api/                # Frontend API client and graph helpers (api.ts)
│   ├── components/         # React components
│   │   ├── editor/         # Editor mode components (NodeAdder, DimensionBuilder, Modals)
│   │   │   ├── CompileModal.tsx
│   │   │   ├── ConstantNodeAdder.tsx
│   │   │   ├── DimensionBuilder.tsx
│   │   │   ├── NodeAdder.tsx
│   │   │   ├── OperationNodeAdder.tsx
│   │   │   ├── RestoreModal.tsx
│   │   │   └── TensorNodeAdder.tsx
│   │   ├── visualizer/     # Visualizer mode controls & inspectors
│   │   │   ├── ColorPicker.tsx
│   │   │   ├── EdgeWindow.tsx
│   │   │   ├── LayoutSelector.tsx
│   │   │   ├── NodeWindow.tsx
│   │   │   ├── ThemeSelector.tsx
│   │   │   └── TransformationOps.tsx
│   │   ├── Cytoscape.tsx   # Cytoscape graph canvas & event handler
│   │   ├── MenuBar.tsx     # Top navigation bar & action toolbar
│   │   └── SidePanel.tsx   # Dynamic side panel (Node adder / Transformation ops)
│   ├── pages/              # Main route views (Home.tsx, App.tsx)
│   ├── routes/             # React Router configuration (MainRoutes.tsx)
│   ├── styles/             # Global CSS and Cytoscape theme stylesheets
│   ├── types/              # TypeScript interface definitions (Cytoscape.ts, Onnx.ts)
│   └── utils/              # Helper utilities (ValueNodeExtractor.ts)
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

---

## 🔌 Backend API Integration

The visualizer connects to an Express server backend running on `http://localhost:3000` (or specified port):

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/files` | `GET` | Retrieve list of available ONNX files |
| `/api/sessions` | `POST` | Initialize a new graph session for a model |
| `/api/sessions/:id` | `DELETE` | Terminate active graph session |
| `/api/sessions/:id/graph` | `GET` | Fetch Cytoscape-formatted graph JSON |
| `/api/sessions/:id/opportunities` | `GET` | List available graph transformation recipes |
| `/api/sessions/:id/apply/:opId` | `POST` | Apply transformation recipe |
| `/api/sessions/:id/undo` | `POST` | Revert last transformation |
| `/api/sessions/:id/redo` | `POST` | Reapply undone transformation |
| `/api/sessions/:id/compile` | `POST` | Compile modified graph back to ONNX format |
| `/api/sessions/:id/export/onnx-json` | `GET` | Download ONNX graph as JSON |
| `/api/sessions/:id/export/unified-json` | `GET` | Download Unified format graph as JSON |

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** (v18 or higher recommended)
- **npm** (v9 or higher)

### Installation

In `onnx-flow` root directory, install dependencies:

```bash
npm install
```

### Running the Application

- **Development Mode** (Frontend + Backend concurrently):
  ```bash
  npm run webapp:dev
  ```
- **Frontend Only**:
  ```bash
  npm run webapp:frontend
  ```
- **Backend Only**:
  ```bash
  npm run webapp:backend
  ```
