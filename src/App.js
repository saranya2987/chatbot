


// App.js (React Frontend)
import React, { useState } from 'react';
import './styles.css';
import axios from 'axios';

const App = () => {
    const [question, setQuestion] = useState('');
    const [botReply, setBotReply] = useState('');
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState('');

    const handleQuestionChange = (e) => {
        setQuestion(e.target.value);
    };

    const handleAskQuestion = async () => {
        console.log("handleAskQuestion called");
        try {
            console.log("sending request");
            const requestBody = JSON.stringify({ natural_language_query: question });
            console.log("Request Body:", requestBody);
            console.log("type of question: ", typeof(question));
    
            const response = await fetch('http://localhost:8000/query/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: requestBody,
            });
    
            if (!response.ok) {
                console.error("API Error:", response.status, response.statusText);
                setBotReply(`API Error: ${response.status} ${response.statusText}`);
                return;
            }
    
            console.log("response received");
            const data = await response.json();
            console.log("response data", data);
    
            if (data.error) {
                setBotReply(`Server Error: ${data.error}`); // differentiate server error
            } else if (data.result && data.result.includes("Movie not found")) {
                setBotReply(data.result);
            } else if (data.result) {
                setBotReply(data.result);
            } else {
                setBotReply("No result from the API");
            }
    
            console.log("bot reply set");
        } catch (error) {
            console.error('Error:', error);
            setBotReply(`Client Error: ${error.message || 'An error occurred. Please try again later.'}`); // differentiate client error.
        }
    };

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
            const response = await axios.post('http://localhost:8000/uploadfile/', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setUploadStatus(response.data.message);
        } catch (error) {
            setUploadStatus("Upload failed.");
            console.error('Upload Error:', error);
        }
    };

    return (
        <div className="file-upload-container">
            <h1>GraphAI</h1>

            <div className="question-area">
                <label htmlFor="question">Ask a Question:</label><br />
                <textarea
                    id="question"
                    value={question}
                    onChange={handleQuestionChange}
                    placeholder="Enter your question here..."
                />
            </div>

            <input type="file" onChange={handleFileChange} />
            <button onClick={handleFileUpload}>Upload</button>

            <button className="ask-button" onClick={handleAskQuestion}>
                Ask
            </button>

            {botReply && <div className="bot-reply">{botReply}</div>}
            {uploadStatus && <div>{uploadStatus}</div>}
        </div>
    );
};

export default App;

