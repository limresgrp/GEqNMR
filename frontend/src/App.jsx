import React, { useState, useEffect } from 'react';

// Base URL for the FastAPI backend
const API_URL = 'http://localhost:8000';

function App() {
  const [backendStatus, setBackendStatus] = useState('Checking connection...');
  const [statusColor, setStatusColor] = useState('text-yellow-500'); 
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState('');
  const [uploadMessageColor, setUploadMessageColor] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [trajectoryFile, setTrajectoryFile] = useState(null);
  const [outputFilePath, setOutputFilePath] = useState(null);
  const [destandardize, setDestandardize] = useState(true); // New state for de-standardization toggle

  // 1. Fetch backend status on component load
  useEffect(() => {
    fetch(API_URL)
      .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
      })
      .then(data => {
        setBackendStatus(data.message || 'Connected, but no message received.');
        setStatusColor('text-green-500');
      })
      .catch(error => {
        console.error('Error fetching backend status:', error);
        setBackendStatus('Connection failed. Is the backend running?');
        setStatusColor('text-red-500');
      });
  }, []); 

  // 2. Handle file selection from the input
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setUploadMessage(''); 
    setTrajectoryFile(null); // Reset trajectory if main file changes
    setOutputFilePath(null); // Clear previous output
  };

  // 3. Handle file upload and API call
  const handleUpload = (event) => {
    event.preventDefault(); 
    
    if (!selectedFile) {
      setUploadMessage('Please select a file first.');
      setUploadMessageColor('text-yellow-500');
      return;
    }

    setIsUploading(true);
    setUploadMessage('Uploading and running workflow...');
    setUploadMessageColor('text-blue-500');
    setOutputFilePath(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    if (trajectoryFile) {
      formData.append('trajectory_file', trajectoryFile);
    }

    formData.append('destandardize', destandardize); // Add the flag to the form data
    
    const targetUrl = `${API_URL}/infer/pdb/`;
    const allowedMimeTypes = ['.xyz', '.pdb', '.gro'];

    // Basic file type check before sending
    const fileExtension = selectedFile.name.split('.').pop().toLowerCase();
    if (!allowedMimeTypes.includes('.' + fileExtension)) { 
        setUploadMessage(`Error: File type .${fileExtension} is not supported.`);
        setUploadMessageColor('text-red-500');
        setIsUploading(false);
        return;
    }

    fetch(targetUrl, {
      method: 'POST',
      body: formData,
    })
    .then(response => {
      const contentType = response.headers.get("content-type");
      if (contentType && contentType.indexOf("application/json") !== -1) {
        return response.json().then(data => ({ ok: response.ok, data }));
      } else {
        return response.text().then(text => {
          throw new Error(text || 'Server returned non-JSON response');
        });
      }
    })
    .then(({ ok, data }) => {
      if (ok) {
        setUploadMessage(`Success! ${data.atoms_predicted} atom predictions generated.`);
        setUploadMessageColor('text-green-500');
        
        // Extract the filename from the full path returned by FastAPI
        const fullPath = data.output_file;
        const filename = fullPath.split('/').pop();
        setOutputFilePath(filename);

      } else {
        throw new Error(data.detail || 'An unknown error occurred');
      }
    })
    .catch(error => {
      console.error('Upload error:', error);
      setUploadMessage(`Error: ${error.message}`);
      setUploadMessageColor('text-red-500');
    })
    .finally(() => {
      setIsUploading(false);
      // Reset the file input visually
      if (document.getElementById('file-upload')) {
          document.getElementById('file-upload').value = '';
      }
      if (document.getElementById('trajectory-upload')) {
          document.getElementById('trajectory-upload').value = '';
      }
    });
  };
  
  const handleTrajectoryFileChange = (event) => {
    setTrajectoryFile(event.target.files[0]);
  };
  const downloadLink = outputFilePath ? `${API_URL}/download/${outputFilePath}` : '#';
  const buttonText = isUploading ? 'Working...' : 'Run Inference';
  const allowedFileTypes = '.pdb, .gro, .xyz';
  const instructions = "Upload a PDB, GRO, or XYZ file. For PDB/GRO, predictions are in the B-factor column. For XYZ, they are in a new 'cs_iso' column. Trajectories are returned as a ZIP of PDBs or a multi-frame XYZ file.";

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center p-8 font-sans">
      <div className="w-full max-w-3xl space-y-8">
        
        {/* Header */}
        <h1 className="text-4xl font-bold text-center text-white">GEqNMR Workflow</h1>

        {/* 1. Backend Status Card */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold mb-3 text-gray-200">Backend Status</h2>
          <div className="flex items-center space-x-3">
            <span className={`h-4 w-4 rounded-full ${
              statusColor === 'text-green-500' ? 'bg-green-500' :
              statusColor === 'text-yellow-500' ? 'bg-yellow-500' : 'bg-red-500'
            }`}></span>
            <p id="status-message" className={`text-lg ${statusColor}`}>
              {backendStatus}
            </p>
          </div>
        </div>

        {/* 2. File Upload Card */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold mb-4 text-gray-200">
            Run NMR Prediction
          </h2>
          <p className="text-gray-400 mb-4">{instructions}</p>
          
          <form onSubmit={handleUpload} className="space-y-4">
            {/* File Input */}
            <div>
              <label htmlFor="file-upload" className="sr-only">Choose file</label>
              <input
                id="file-upload"
                type="file"
                onChange={handleFileChange}
                accept={allowedFileTypes}
                className="block w-full text-sm text-gray-400
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-indigo-600 file:text-white
                  hover:file:bg-indigo-700
                  disabled:opacity-50"
                disabled={isUploading}
              />
            </div>

            {/* Trajectory File Input (Optional) */}
            {selectedFile && (
              <div>
                <label htmlFor="trajectory-upload" className="block text-sm font-medium text-gray-400 mb-1">
                  Trajectory File (Optional)
                </label>
                <input
                  id="trajectory-upload"
                  type="file"
                  onChange={handleTrajectoryFileChange}
                  className="block w-full text-sm text-gray-400
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-md file:border-0
                    file:text-sm file:font-semibold
                    file:bg-gray-600 file:text-white
                    hover:file:bg-gray-700 disabled:opacity-50"
                  disabled={isUploading} />
              </div>
            )}
            {/* NEW: De-standardization Toggle for Inference Mode */}
            {(
              <div className="flex items-center justify-center pt-2">
                <input
                  id="destandardize-checkbox"
                  type="checkbox"
                  checked={destandardize}
                  onChange={(e) => setDestandardize(e.target.checked)}
                  className="w-4 h-4 text-indigo-600 bg-gray-700 border-gray-600 rounded focus:ring-indigo-500"
                />
                <label htmlFor="destandardize-checkbox" className="ml-2 text-sm font-medium text-gray-300">
                  De-standardize predictions before saving
                </label>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isUploading || !selectedFile}
              className="w-full bg-green-600 text-white font-bold py-3 px-4 rounded-md
                hover:bg-green-700
                focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50
                disabled:bg-gray-600 disabled:cursor-not-allowed"
            >
              {buttonText}
            </button>
          </form>

          {/* Status and Download Message */}
          {uploadMessage && (
            <div id="status-message" className={`mt-4 text-center font-medium ${uploadMessageColor}`}>
              {uploadMessage}
            </div>
          )}
          
          {outputFilePath && (
              <div className="mt-6 text-center">
                  <a
                      href={downloadLink}
                      download={outputFilePath}
                      className="inline-flex items-center justify-center bg-blue-600 text-white font-bold py-2 px-6 rounded-md
                          hover:bg-blue-700 transition-colors duration-200"
                  >
                      {/* Icon for download */}
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                      Download Output: {outputFilePath}
                  </a>
                  <p className="text-sm text-gray-500 mt-2">
                      ({outputFilePath.endsWith('.zip') ? 'ZIP archive of PDB frames.' : (outputFilePath.endsWith('.xyz') ? "Extended XYZ file with 'cs_iso' column." : 'Predictions are in the B-factor column.')})
                  </p>
              </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default App;
