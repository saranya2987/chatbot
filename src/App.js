import React, { useState } from 'react';
import './styles.css';

const App = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState(null);
    const [question, setQuestion] = useState('');
    const [botReply, setBotReply] = useState('');

    const handleFileChange = (e) => {
        setSelectedFile(e.target.files[0]);
    };

    const handleFileUpload = async () => {
        if (!selectedFile) {
            alert("Please select a file to upload.");
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            setUploadStatus("Uploading...");

            const uploadResponse = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            if (!uploadResponse.ok) {
                try {
                    const errorResponseClone = uploadResponse.clone();
                    const errorData = await errorResponseClone.json();
                    setUploadStatus(errorData.message || uploadResponse.statusText);
                    console.error("File upload error:", errorData || uploadResponse.statusText);
                } catch (jsonError) {
                    try {
                        const errorText = await uploadResponse.text();
                        setUploadStatus(uploadResponse.statusText);
                        console.error("File upload error (HTML):", uploadResponse.statusText, errorText);
                    } catch (textError) {
                        setUploadStatus("An error occurred during upload.");
                        console.error("File upload error: Could not read response text", textError);
                    }
                }
                return;
            }

            setUploadStatus("File uploaded and processed successfully!");
            setSelectedFile(null);
            setQuestion('');
            setBotReply('');

        } catch (error) {
            setUploadStatus(`An error occurred: ${error.message}`);
            console.error("General error:", error);
        }
    };

    const handleQuestionChange = (e) => {
        setQuestion(e.target.value);
    };

    const openFile = () => {
        if (selectedFile) {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(selectedFile);
            link.download = selectedFile.name;
            link.click();
        } else {
            alert("No file selected to download.");
        }
    };

    const handleAskQuestion = async () => {
        try {
            const response = await fetch('/api/chat', { // Corrected URL, relative path
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question }),
            });
            const data = await response.json();
            setBotReply(data.reply); // Only display the reply

        } catch (error) {
            console.error('Error:', error);
            setBotReply('An error occurred.');
        }
    };

    return (
        <div className="file-upload-container">
            <h1>GraphAI</h1>

            <div className="question-area">
                <label htmlFor="question">Ask a Question (Optional):</label><br />
                <textarea
                    id="question"
                    value={question}
                    onChange={handleQuestionChange}
                    placeholder="Enter your question here..."
                />
            </div>

            <input type="file" id="fileInput" onChange={handleFileChange} />
            <label htmlFor="fileInput" className="file-input-label">
                Choose File
            </label>

            {selectedFile && (
                <div className="file-info">
                    Selected File: {selectedFile.name}
                    <button onClick={openFile} className="open-button">Download File</button>
                </div>
            )}

            <button onClick={handleFileUpload} disabled={!selectedFile} className="upload-button">
                Upload
            </button>

            <button className="ask-button" onClick={handleAskQuestion}>
                Ask
            </button>

            {uploadStatus && <div className="upload-status">{uploadStatus}</div>}
            {botReply && <div className="bot-reply">{botReply}</div>}
        </div>
    );
};

export default App;

// // App.js (Frontend)
// import React, { useState } from 'react';
// import './styles.css';

// const App = () => {
//     const [selectedFile, setSelectedFile] = useState(null);
//     const [uploadStatus, setUploadStatus] = useState(null);
//     const [question, setQuestion] = useState('');
//     const [botReply, setBotReply] = useState('');

//     const handleFileChange = (e) => {
//         setSelectedFile(e.target.files[0]);
//     };

//     const handleFileUpload = async () => {
//         if (!selectedFile) {
//             alert("Please select a file to upload.");
//             return;
//         }

//         const formData = new FormData();
//         formData.append("file", selectedFile);

//         try {
//             setUploadStatus("Uploading...");

//             const uploadResponse = await fetch("/api/upload", {
//                 method: "POST",
//                 body: formData,
//             });

//             if (!uploadResponse.ok) {
//                 try {
//                     const errorResponseClone = uploadResponse.clone();
//                     const errorData = await errorResponseClone.json();
//                     setUploadStatus(errorData.message || uploadResponse.statusText);
//                     console.error("File upload error:", errorData || uploadResponse.statusText);
//                 } catch (jsonError) {
//                     try {
//                         const errorText = await uploadResponse.text();
//                         setUploadStatus(uploadResponse.statusText);
//                         console.error("File upload error (HTML):", uploadResponse.statusText, errorText);
//                     } catch (textError) {
//                         setUploadStatus("An error occurred during upload.");
//                         console.error("File upload error: Could not read response text", textError);
//                     }
//                 }
//                 return;
//             }

//             // const uploadResult = await uploadResponse.json();
//             // const uploadedFileName = uploadResult.filename;

//             setUploadStatus("File uploaded and processed successfully!");
//             setSelectedFile(null);
//             setQuestion('');
//             setBotReply('');

//         } catch (error) {
//             setUploadStatus(`An error occurred: ${error.message}`);
//             console.error("General error:", error);
//         }
//     };

//     const handleQuestionChange = (e) => {
//         setQuestion(e.target.value);
//     };

//     const openFile = () => {
//         if (selectedFile) {
//             const link = document.createElement('a');
//             link.href = URL.createObjectURL(selectedFile);
//             link.download = selectedFile.name;
//             link.click();
//         } else {
//             alert("No file selected to download.");
//         }
//     };

//     const handleAskQuestion = async () => {
//         try {
//             const response = await fetch('/api/chat', { // Corrected URL, relative path
//                 method: 'POST',
//                 headers: { 'Content-Type': 'application/json' },
//                 body: JSON.stringify({ question }),
//             });
//             const data = await response.json();
//             setBotReply(data.reply);
//         } catch (error) {
//             console.error('Error:', error);
//             setBotReply('An error occurred.');
//         }
//     };

//     return (
//         <div className="file-upload-container">
//             <h1>GraphAI</h1>

//             <div className="question-area">
//                 <label htmlFor="question">Ask a Question (Optional):</label><br />
//                 <textarea
//                     id="question"
//                     value={question}
//                     onChange={handleQuestionChange}
//                     placeholder="Enter your question here..."
//                 />
//             </div>

//             <input type="file" id="fileInput" onChange={handleFileChange} />
//             <label htmlFor="fileInput" className="file-input-label">
//                 Choose File
//             </label>

//             {selectedFile && (
//                 <div className="file-info">
//                     Selected File: {selectedFile.name}
//                     <button onClick={openFile} className="open-button">Download File</button>
//                 </div>
//             )}

//             <button onClick={handleFileUpload} disabled={!selectedFile} className="upload-button">
//                 Upload
//             </button>

//             <button className="ask-button" onClick={handleAskQuestion}>
//                 Ask
//             </button>

//             {uploadStatus && <div className="upload-status">{uploadStatus}</div>}
//             {botReply && <div className="bot-reply">{botReply}</div>}
//         </div>
//     );
// };

// export default App;