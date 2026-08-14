/**
 * @file Home.tsx
 * @description Homepage and Model File Explorer component. Fetches available ONNX model files
 * from the backend server, provides searching, sorting (by name, size, last modified date),
 * ascending/descending order toggling, and session initialization navigation.
 */

import { useEffect, useMemo, useState } from "react";
import Select from "react-select";
import { getAvailableFiles, startNewSession, type ServerFileInfo } from "../api/api";
import { Link } from "react-router-dom";

/**
 * Filter options for sorting the available ONNX files list.
 */
const filterOptions = [
    { label: "Name", value: "name" },
    { label: "Size", value: "size" },
    { label: "Last Modified", value: "lastModified" },
];

/**
 * Represents file descriptor metadata for an ONNX model file on the server.
 */
interface OnnxFile {
    /** File name including extension */
    name: string;
    /** File size in bytes */
    size: number;
    /** ISO 8601 modification timestamp */
    lastModified: string;
}

/**
 * Home page component that displays the list of available ONNX files and handles
 * search filtering, sorting, and session launching.
 *
 * @returns JSX element for the homepage view
 */
function Home() {
    const [onnxFiles, setOnnxFiles] = useState<OnnxFile[]>([]);
    const [filterOption, setFilterOption] = useState<string>(filterOptions[0].value);
    const [orderOption, setOrderOption] = useState<boolean>(true); // true for ascending, false for descending
    const [searchTerm, setSearchTerm] = useState<string>("");

    /**
     * Initial data fetching effect: retrieves the list of ONNX files from the backend server on mount.
     */
    useEffect(() => {
        getAvailableFiles()
            .then((files: ServerFileInfo[]) => {
                setOnnxFiles(files);
            })
            .catch((error) => {
                console.error("Error fetching ONNX files:", error);
            });
    }, []);

    /**
     * Memoized list of displayed files, filtered by search query and sorted by the active filter criteria and order.
     */
    const displayedFiles = useMemo(() => {
        const sorted = [...onnxFiles].sort((a, b) => {
            let result;
            switch (filterOption) {
                case "name":
                    result = a.name.localeCompare(b.name, undefined, {
                        sensitivity: "base",
                        numeric: true,
                    });
                    break;
                case "size":
                    result = a.size - b.size;
                    break;
                case "lastModified":
                    result = (
                        new Date(a.lastModified).getTime() -
                        new Date(b.lastModified).getTime()
                    );
                    break;
                default:
                    return 0;
            }
            return orderOption ? result : -result;
        });

        return sorted.filter((file) =>
            file.name.toLowerCase().includes(searchTerm.toLowerCase())
        );
    }, [onnxFiles, filterOption, searchTerm, orderOption]);

    return (
        <>
        <header>
            <h1>ONNX Graphical Interface Homepage</h1>
        </header>
        <main>
            <button onClick={() => {
                setOnnxFiles([]);
                getAvailableFiles()
                    .then((files: ServerFileInfo[]) => {
                        setOnnxFiles(files);
                    })
                    .catch((error) => {
                        console.error("Error fetching ONNX files:", error);
                    });
            }}>
                Refresh
            </button>
            <button onClick={() => setOrderOption(!orderOption)}>
                {orderOption ? "Ascending" : "Descending"}
            </button>
            <Select 
                options={filterOptions}
                placeholder="Filter by..."
                defaultValue={filterOptions[0]}
                onChange={(selectedOption) => setFilterOption(selectedOption!.value)}
            />
            <input
                type="text"
                placeholder="Search for ONNX files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
            />
            {onnxFiles.length != 0 ? (
                <ul style={{
                     overflow: "auto",
                     maxHeight: "80vh" 
                }}>
                    {displayedFiles.map((file, index) => (
                        <li key={index}>
                                <h2 style={{cursor: "pointer"}}>{file.name}</h2>
                                <p>Size: {file.size} bytes</p>
                                <p>Last Modified: {(new Date(file.lastModified)).toLocaleString()}</p>
                                <Link to={`/app/${file.name}`} target="_blank" rel="noopener noreferrer" onClick={() => startNewSession(3000, file.name)}>Open File</Link>
                            </li>
                    ))}
                </ul>
            ) : (
            <p>
                Loading...
            </p>
            )
            }
        </main>
        </>
    );
}

export default Home;