import { useEffect, useMemo, useState } from "react";
import Select from "react-select";
import { getAvailableFiles } from "../api/api";

const filterOptions = [
    {label: "Name", value: "name"},
    {label: "Size", value: "size"},
    {label: "Last Modified", value: "lastModified"},
];

interface OnnxFile {
    name: string;
    size: number; // in bytes
    lastModified: string; // ISO 8601 timestamp
}

function Home() {
    const [onnxFiles, setOnnxFiles] = useState<OnnxFile[]>([]);
    const [filterOption, setFilterOption] = useState<string>(filterOptions[0].value);
    const [orderOption, setOrderOption] = useState<boolean>(true); // true for ascending, false for descending
    const [searchTerm, setSearchTerm] = useState<string>("");

    useEffect(() => {
        getAvailableFiles()
            .then((files) => {
                setOnnxFiles(files);
            })
            .catch((error) => {
                console.error("Error fetching ONNX files:", error);
            });
    }, []);

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
                    .then((files) => {
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
                <ul style={{ overflow: "auto" }}>
                    {displayedFiles.map((file, index) => (
                        <li key={index}>
                                <h2 style={{cursor: "pointer"}}>{file.name}</h2>
                                <p>Size: {file.size} bytes</p>
                                <p>Last Modified: {(new Date(file.lastModified)).toLocaleString("pt-PT")}</p>
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