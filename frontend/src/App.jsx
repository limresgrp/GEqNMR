import React, { useState, useEffect, useMemo } from 'react';

// Base URL for the FastAPI backend
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8273';

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
  const [modelOptions, setModelOptions] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelLoadError, setModelLoadError] = useState('');
  const [view, setView] = useState('upload');
  const [latestResult, setLatestResult] = useState('');
  const [highlightedResult, setHighlightedResult] = useState('');
  const [results, setResults] = useState([]);
  const [isLoadingResults, setIsLoadingResults] = useState(false);
  const [resultsError, setResultsError] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [progressPhase, setProgressPhase] = useState('idle');
  const [preparedData, setPreparedData] = useState(null);
  const [preparedInputs, setPreparedInputs] = useState([]);
  const [selectedPreparedId, setSelectedPreparedId] = useState('');
  const [preparedName, setPreparedName] = useState('');
  const [prepareProgressMessage, setPrepareProgressMessage] = useState('');
  const [frameSlice, setFrameSlice] = useState('');
  const [selectedKey, setSelectedKey] = useState('');
  const [prepareError, setPrepareError] = useState('');
  const [selectedBatch, setSelectedBatch] = useState(0);
  const [activeKeyDetails, setActiveKeyDetails] = useState(null);
  const [keyDetailsError, setKeyDetailsError] = useState('');
  const [latticeMatrix, setLatticeMatrix] = useState([
    ['1', '0', '0'],
    ['0', '1', '0'],
    ['0', '0', '1'],
  ]);
  const [latticeDetails, setLatticeDetails] = useState(null);
  const [latticeError, setLatticeError] = useState('');
  const [latticeSaving, setLatticeSaving] = useState(false);
  const [predictionResults, setPredictionResults] = useState([]);
  const [predictionResultsError, setPredictionResultsError] = useState('');
  const [isLoadingPredictionResults, setIsLoadingPredictionResults] = useState(false);
  const [selectedPredictionResults, setSelectedPredictionResults] = useState([]);
  const [predictionDataByResult, setPredictionDataByResult] = useState({});
  const [visualizationMode, setVisualizationMode] = useState('violin');
  const [selectedAtomIds, setSelectedAtomIds] = useState([]);
  const [selectedPairAtomIds, setSelectedPairAtomIds] = useState([]);

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

  const fetchResults = () => {
    setIsLoadingResults(true);
    setResultsError('');
    fetch(`${API_URL}/results`)
      .then(response => {
        if (!response.ok) throw new Error('Failed to load results');
        return response.json();
      })
      .then(data => {
        const list = Array.isArray(data.results) ? data.results : [];
        setResults(list);
        setIsLoadingResults(false);
      })
      .catch(error => {
        console.error('Error fetching results:', error);
        setResults([]);
        setResultsError('Unable to load results from the backend.');
        setIsLoadingResults(false);
      });
  };

  const fetchPreparedInputs = () => {
    fetch(`${API_URL}/prepared`)
      .then(response => {
        if (!response.ok) throw new Error('Failed to load prepared inputs');
        return response.json();
      })
      .then(data => {
        setPreparedInputs(Array.isArray(data.prepared) ? data.prepared : []);
      })
      .catch(error => {
        console.error('Prepared input list error:', error);
        setPrepareError('Unable to load stored prepared inputs.');
      });
  };

  const openPreparedInput = (preparedId) => {
    if (!preparedId) {
      setPreparedData(null);
      setSelectedPreparedId('');
      return;
    }
    setPrepareError('');
    fetch(`${API_URL}/prepared/${encodeURIComponent(preparedId)}`)
      .then(response => {
        if (!response.ok) throw new Error('Failed to load prepared input');
        return response.json();
      })
      .then(data => {
        setPreparedData(data);
        setSelectedPreparedId(data.id);
        setSelectedKey(data.keys?.[0]?.name || '');
        setSelectedBatch(0);
        setActiveKeyDetails(null);
        setKeyDetailsError('');
        setLatticeDetails(null);
        setView('prepare');
      })
      .catch(error => {
        console.error('Prepared detail error:', error);
        setPrepareError('Unable to open the selected prepared input.');
      });
  };

  const deletePreparedInput = (preparedId) => {
    fetch(`${API_URL}/prepared/${encodeURIComponent(preparedId)}`, { method: 'DELETE' })
      .then(response => {
        if (!response.ok) throw new Error('Failed to delete prepared input');
        if (selectedPreparedId === preparedId) {
          setPreparedData(null);
          setSelectedPreparedId('');
          setSelectedKey('');
        }
        fetchPreparedInputs();
      })
      .catch(error => {
        console.error('Prepared delete error:', error);
        setPrepareError('Unable to delete the selected prepared input.');
      });
  };

  const pollPrepareJob = (jobId) => {
    fetch(`${API_URL}/prepare/jobs/${jobId}`)
      .then(response => {
        if (!response.ok) throw new Error('Failed to load prepare job');
        return response.json();
      })
      .then(data => {
        setUploadProgress(Math.round((data.progress || 0) * 100));
        setPrepareProgressMessage(data.message || '');
        if (data.status === 'completed') {
          setIsUploading(false);
          setProgressPhase('idle');
          setUploadMessage('Input prepared and stored.');
          setUploadMessageColor('text-green-500');
          setPreparedData(data.result);
          setSelectedPreparedId(data.result?.id || '');
          setSelectedKey(data.result?.keys?.[0]?.name || '');
          setSelectedBatch(0);
          fetchPreparedInputs();
          setView('prepare');
          return;
        }
        if (data.status === 'failed') {
          throw new Error(data.error || data.message || 'Preparation failed');
        }
        setTimeout(() => pollPrepareJob(jobId), 800);
      })
      .catch(error => {
        console.error('Prepare job error:', error);
        setIsUploading(false);
        setProgressPhase('idle');
        setUploadMessage(`Error: ${error.message}`);
        setUploadMessageColor('text-red-500');
        setPrepareError('Unable to prepare input data.');
      });
  };

  const fetchPredictionResults = () => {
    setIsLoadingPredictionResults(true);
    setPredictionResultsError('');
    fetch(`${API_URL}/prediction-results`)
      .then(response => {
        if (!response.ok) throw new Error('Failed to load prediction results');
        return response.json();
      })
      .then(data => {
        const list = Array.isArray(data.results) ? data.results : [];
        setPredictionResults(list);
        setIsLoadingPredictionResults(false);
      })
      .catch(error => {
        console.error('Prediction result load error:', error);
        setPredictionResults([]);
        setPredictionResultsError('Unable to load prediction result metadata.');
        setIsLoadingPredictionResults(false);
      });
  };

  // 1b. Fetch available models for selection
  useEffect(() => {
    fetch(`${API_URL}/models`)
      .then(response => {
        if (!response.ok) throw new Error('Failed to load models');
        return response.json();
      })
      .then(data => {
        const models = Array.isArray(data.models) ? data.models : [];
        setModelOptions(models);
        const defaultModel = data.default_model || models[0] || '';
        setSelectedModel(defaultModel);
        setModelLoadError('');
      })
      .catch(error => {
        console.error('Error fetching models:', error);
        setModelOptions([]);
        setSelectedModel('');
        setModelLoadError('Unable to load models from the backend.');
      });
  }, []);

  useEffect(() => {
    if (view === 'results') {
      fetchResults();
    }
    if (view === 'prepare') {
      fetchPreparedInputs();
    }
    if (view === 'visualize') {
      fetchPredictionResults();
    }
  }, [view]);

  useEffect(() => {
    selectedPredictionResults.forEach((resultName) => {
      if (predictionDataByResult[resultName]) {
        return;
      }
      fetch(`${API_URL}/prediction-results/${encodeURIComponent(resultName)}`)
        .then(response => {
          if (!response.ok) throw new Error(`Failed to load ${resultName}`);
          return response.json();
        })
        .then(data => {
          setPredictionDataByResult((prev) => ({ ...prev, [resultName]: data }));
        })
        .catch(error => {
          console.error('Prediction metadata error:', error);
          setPredictionResultsError(`Unable to load prediction metadata for ${resultName}.`);
        });
    });
  }, [selectedPredictionResults, predictionDataByResult]);

  const fetchKeyDetails = (keyName, batchIndex) => {
    if (!preparedData?.id || !keyName) {
      return;
    }
    const keyMeta = preparedData.keys?.find((key) => key.name === keyName);
    const hasBatchDimension = keyMeta && preparedData.num_molecules
      && Array.isArray(keyMeta.shape)
      && keyMeta.shape[0] === preparedData.num_molecules;
    const params = new URLSearchParams();
    if (hasBatchDimension) {
      params.set('batch_index', batchIndex.toString());
    }

    setKeyDetailsError('');
    fetch(`${API_URL}/prepare/${preparedData.id}/keys/${encodeURIComponent(keyName)}${params.toString() ? `?${params}` : ''}`)
      .then(response => {
        if (!response.ok) throw new Error('Failed to load key details');
        return response.json();
      })
      .then(data => {
        setActiveKeyDetails(data.key || null);
      })
      .catch(error => {
        console.error('Key detail error:', error);
        setActiveKeyDetails(null);
        setKeyDetailsError('Unable to load key details.');
      });
  };

  const fetchLatticeDetails = (batchIndex) => {
    if (!preparedData?.id) {
      return;
    }
    const hasLattice = preparedData.keys?.some((key) => key.name === 'Lattice');
    if (!hasLattice) {
      setLatticeDetails(null);
      return;
    }

    const params = new URLSearchParams();
    if (preparedData.num_molecules > 0) {
      params.set('batch_index', batchIndex.toString());
    }

    fetch(`${API_URL}/prepare/${preparedData.id}/keys/Lattice?${params}`)
      .then(response => {
        if (!response.ok) throw new Error('Failed to load lattice details');
        return response.json();
      })
      .then(data => {
        setLatticeDetails(data.key || null);
      })
      .catch(error => {
        console.error('Lattice detail error:', error);
        setLatticeDetails(null);
      });
  };

  const openResults = () => {
    setHighlightedResult('');
    setView('results');
  };

  const goToResults = () => {
    if (latestResult) {
      setHighlightedResult(latestResult);
    }
    setView('results');
  };

  const formatBytes = (bytes) => {
    if (!Number.isFinite(bytes)) return 'Unknown size';
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex += 1;
    }
    return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
  };

  const formatTimestamp = (timestampSeconds) => {
    if (!Number.isFinite(timestampSeconds)) return 'Unknown time';
    return new Date(timestampSeconds * 1000).toLocaleString();
  };

  useEffect(() => {
    const latticeSource = latticeDetails?.matrix
      || (activeKeyDetails?.name === 'Lattice' ? activeKeyDetails.matrix : null);
    if (latticeSource && Array.isArray(latticeSource)) {
      const nextMatrix = latticeSource.map((row) =>
        row.map((value) => (Number.isFinite(value) ? value.toString() : '0'))
      );
      if (nextMatrix.length === 3 && nextMatrix.every((row) => row.length === 3)) {
        setLatticeMatrix(nextMatrix);
        return;
      }
    }
    if (preparedData && !preparedData.keys?.some((key) => key.name === 'Lattice')) {
      setLatticeMatrix([
        ['1', '0', '0'],
        ['0', '1', '0'],
        ['0', '0', '1'],
      ]);
    }
  }, [latticeDetails, activeKeyDetails, preparedData]);

  const activeKey = activeKeyDetails || preparedData?.keys?.find((key) => key.name === selectedKey);
  const selectedPredictionData = selectedPredictionResults
    .map((resultName) => predictionDataByResult[resultName])
    .filter(Boolean);
  const resultColors = ['#34d399', '#60a5fa', '#f59e0b', '#f472b6', '#a78bfa', '#22d3ee', '#fb7185', '#c084fc'];
  const atomOptions = useMemo(() => {
    const seen = new Map();
    selectedPredictionData.forEach((result) => {
      (result.atoms || []).forEach((atom) => {
        if (!seen.has(atom.id)) {
          seen.set(atom.id, { id: atom.id, label: atom.label || atom.id });
        }
      });
    });
    return Array.from(seen.values()).sort((a, b) => a.label.localeCompare(b.label));
  }, [selectedPredictionData]);

  const getResultAtomValues = (result, atomId) => {
    const atom = result?.atoms?.find((entry) => entry.id === atomId);
    return Array.isArray(atom?.values) ? atom.values.filter((value) => Number.isFinite(value)) : [];
  };

  const renderViolinPlot = () => {
    const selectedAtoms = atomOptions.filter((atom) => selectedAtomIds.includes(atom.id));
    if (selectedPredictionData.length === 0 || selectedAtoms.length === 0) {
      return <div className="text-sm text-gray-400">Select at least one result and one atom.</div>;
    }

    const series = [];
    selectedAtoms.forEach((atom) => {
      selectedPredictionData.forEach((result, resultIndex) => {
        const values = getResultAtomValues(result, atom.id);
        if (values.length > 0) {
          series.push({ atom, result, resultIndex, values });
        }
      });
    });
    if (series.length === 0) {
      return <div className="text-sm text-gray-400">No prediction values available for the selected atoms.</div>;
    }

    const allValues = series.flatMap((item) => item.values);
    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);
    const yMin = minValue === maxValue ? minValue - 1 : minValue;
    const yMax = minValue === maxValue ? maxValue + 1 : maxValue;
    const width = Math.max(760, selectedAtoms.length * 130);
    const height = 420;
    const margin = { top: 24, right: 24, bottom: 92, left: 62 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const yScale = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * plotHeight;
    const groupWidth = plotWidth / selectedAtoms.length;
    const bins = 28;

    return (
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="min-w-full h-[420px]">
          <rect x="0" y="0" width={width} height={height} fill="#111827" />
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const value = yMin + (yMax - yMin) * tick;
            const y = yScale(value);
            return (
              <g key={`y-${tick}`}>
                <line x1={margin.left} x2={width - margin.right} y1={y} y2={y} stroke="#374151" strokeWidth="1" />
                <text x={margin.left - 10} y={y + 4} textAnchor="end" fill="#9ca3af" fontSize="11">
                  {value.toFixed(1)}
                </text>
              </g>
            );
          })}
          {series.map((item) => {
            const atomIndex = selectedAtoms.findIndex((atom) => atom.id === item.atom.id);
            const resultCount = selectedPredictionData.length;
            const centerX = margin.left + atomIndex * groupWidth + groupWidth / 2
              + (item.resultIndex - (resultCount - 1) / 2) * Math.min(24, groupWidth / Math.max(resultCount, 1));
            const color = resultColors[item.resultIndex % resultColors.length];
            const counts = Array.from({ length: bins }, () => 0);
            item.values.forEach((value) => {
              const rawIndex = Math.floor(((value - yMin) / (yMax - yMin)) * bins);
              const binIndex = Math.min(Math.max(rawIndex, 0), bins - 1);
              counts[binIndex] += 1;
            });
            const maxCount = Math.max(...counts, 1);
            const halfWidth = Math.min(26, groupWidth / Math.max(resultCount * 2.5, 3));
            const leftPoints = counts.map((count, binIndex) => {
              const value = yMin + ((binIndex + 0.5) / bins) * (yMax - yMin);
              const x = centerX - (count / maxCount) * halfWidth;
              return `${x},${yScale(value)}`;
            });
            const rightPoints = counts.slice().reverse().map((count, reverseIndex) => {
              const binIndex = bins - 1 - reverseIndex;
              const value = yMin + ((binIndex + 0.5) / bins) * (yMax - yMin);
              const x = centerX + (count / maxCount) * halfWidth;
              return `${x},${yScale(value)}`;
            });
            const mean = item.values.reduce((sum, value) => sum + value, 0) / item.values.length;
            return (
              <g key={`${item.result.result}-${item.atom.id}`}>
                <polygon points={[...leftPoints, ...rightPoints].join(' ')} fill={color} opacity="0.42" stroke={color} strokeWidth="1.5" />
                <line x1={centerX - halfWidth} x2={centerX + halfWidth} y1={yScale(mean)} y2={yScale(mean)} stroke={color} strokeWidth="2" />
              </g>
            );
          })}
          {selectedAtoms.map((atom, atomIndex) => {
            const x = margin.left + atomIndex * groupWidth + groupWidth / 2;
            return (
              <text key={atom.id} x={x} y={height - 46} textAnchor="end" fill="#d1d5db" fontSize="11" transform={`rotate(-35 ${x} ${height - 46})`}>
                {atom.label}
              </text>
            );
          })}
          <text x={18} y={height / 2} fill="#d1d5db" fontSize="12" transform={`rotate(-90 18 ${height / 2})`}>
            Predicted cs
          </text>
        </svg>
        <div className="flex flex-wrap gap-3 mt-3">
          {selectedPredictionData.map((result, index) => (
            <div key={result.result} className="flex items-center gap-2 text-xs text-gray-300">
              <span className="h-3 w-3 rounded-full" style={{ backgroundColor: resultColors[index % resultColors.length] }} />
              <span>{result.result}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderPairPlot = () => {
    if (selectedPredictionData.length === 0 || selectedPairAtomIds.length !== 2) {
      return <div className="text-sm text-gray-400">Select at least one result and exactly two atoms.</div>;
    }
    const [xAtomId, yAtomId] = selectedPairAtomIds;
    const xAtom = atomOptions.find((atom) => atom.id === xAtomId);
    const yAtom = atomOptions.find((atom) => atom.id === yAtomId);
    const points = [];
    selectedPredictionData.forEach((result, resultIndex) => {
      const xs = getResultAtomValues(result, xAtomId);
      const ys = getResultAtomValues(result, yAtomId);
      const count = Math.min(xs.length, ys.length);
      for (let index = 0; index < count; index += 1) {
        points.push({ x: xs[index], y: ys[index], result, resultIndex });
      }
    });
    if (points.length === 0) {
      return <div className="text-sm text-gray-400">No paired values available for the selected atoms.</div>;
    }
    const xValues = points.map((point) => point.x);
    const yValues = points.map((point) => point.y);
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);
    const minY = Math.min(...yValues);
    const maxY = Math.max(...yValues);
    const xMin = minX === maxX ? minX - 1 : minX;
    const xMax = minX === maxX ? maxX + 1 : maxX;
    const yMin = minY === maxY ? minY - 1 : minY;
    const yMax = minY === maxY ? maxY + 1 : maxY;
    const width = 760;
    const height = 460;
    const margin = { top: 24, right: 28, bottom: 72, left: 70 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const xScale = (value) => margin.left + ((value - xMin) / (xMax - xMin)) * plotWidth;
    const yScale = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * plotHeight;
    return (
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="min-w-full h-[460px]">
          <rect x="0" y="0" width={width} height={height} fill="#111827" />
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const xValue = xMin + (xMax - xMin) * tick;
            const yValue = yMin + (yMax - yMin) * tick;
            const x = xScale(xValue);
            const y = yScale(yValue);
            return (
              <g key={`grid-${tick}`}>
                <line x1={x} x2={x} y1={margin.top} y2={height - margin.bottom} stroke="#374151" strokeWidth="1" />
                <line x1={margin.left} x2={width - margin.right} y1={y} y2={y} stroke="#374151" strokeWidth="1" />
                <text x={x} y={height - margin.bottom + 20} textAnchor="middle" fill="#9ca3af" fontSize="11">{xValue.toFixed(1)}</text>
                <text x={margin.left - 10} y={y + 4} textAnchor="end" fill="#9ca3af" fontSize="11">{yValue.toFixed(1)}</text>
              </g>
            );
          })}
          {points.map((point, index) => (
            <circle
              key={`${point.result.result}-${index}`}
              cx={xScale(point.x)}
              cy={yScale(point.y)}
              r="3.5"
              fill={resultColors[point.resultIndex % resultColors.length]}
              opacity="0.48"
            />
          ))}
          <text x={margin.left + plotWidth / 2} y={height - 24} textAnchor="middle" fill="#d1d5db" fontSize="12">
            {xAtom?.label || xAtomId}
          </text>
          <text x={18} y={margin.top + plotHeight / 2} fill="#d1d5db" fontSize="12" transform={`rotate(-90 18 ${margin.top + plotHeight / 2})`}>
            {yAtom?.label || yAtomId}
          </text>
        </svg>
      </div>
    );
  };

  const renderHistogram = (histogram, isBoolean) => {
    if (!histogram || !Array.isArray(histogram.counts) || histogram.counts.length === 0) {
      return <p className="text-sm text-gray-500">No distribution available.</p>;
    }
    if (isBoolean && histogram.counts.length === 2) {
      const [falseCount, trueCount] = histogram.counts;
      const maxCount = Math.max(falseCount, trueCount, 1);
      const falseWidth = Math.round((falseCount / maxCount) * 100);
      const trueWidth = Math.round((trueCount / maxCount) * 100);
      return (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="w-20 text-xs text-gray-400">False</div>
            <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div className="h-full bg-red-400/80" style={{ width: `${falseWidth}%` }} />
            </div>
            <div className="w-12 text-xs text-gray-400 text-right">{falseCount}</div>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-20 text-xs text-gray-400">True</div>
            <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div className="h-full bg-green-400/80" style={{ width: `${trueWidth}%` }} />
            </div>
            <div className="w-12 text-xs text-gray-400 text-right">{trueCount}</div>
          </div>
        </div>
      );
    }
    const maxCount = Math.max(...histogram.counts, 1);
    const bins = histogram.bins || [];
    return (
      <div className="space-y-2">
        {histogram.counts.map((count, index) => {
          const width = Math.round((count / maxCount) * 100);
          const rangeLabel = bins.length > index + 1
            ? `${bins[index]} to ${bins[index + 1]}`
            : `Bin ${index + 1}`;
          return (
            <div key={`bin-${index}`} className="flex items-center gap-3">
              <div className="w-28 text-xs text-gray-400 truncate">{rangeLabel}</div>
              <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-green-400/80" style={{ width: `${width}%` }} />
              </div>
              <div className="w-12 text-xs text-gray-400 text-right">{count}</div>
            </div>
          );
        })}
      </div>
    );
  };

  const renderLatticePreview = (matrixValues) => {
    if (!matrixValues || matrixValues.length !== 3) {
      return <p className="text-sm text-gray-500">No lattice data available.</p>;
    }

    const vectors = matrixValues.map((row) => row.map((value) => parseFloat(value)));
    const matrixValid = vectors.every((row) => row.length === 3 && row.every((value) => Number.isFinite(value)));
    if (!matrixValid) {
      return <p className="text-sm text-gray-500">Enter numeric lattice values to preview.</p>;
    }

    const flattened = vectors.flat();
    if (flattened.every((value) => Math.abs(value) < 1e-6)) {
      return <p className="text-sm text-gray-500">Lattice vectors are all zero.</p>;
    }

    const [a, b, c] = vectors;
    const points3d = [
      [0, 0, 0],
      a,
      b,
      c,
      [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
      [a[0] + c[0], a[1] + c[1], a[2] + c[2]],
      [b[0] + c[0], b[1] + c[1], b[2] + c[2]],
      [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]],
    ];

    const project = (point) => {
      const [x, y, z] = point;
      return [x - y, (x + y) / 2 - z];
    };

    const points2d = points3d.map(project);
    const xs = points2d.map((point) => point[0]);
    const ys = points2d.map((point) => point[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const width = Math.max(maxX - minX, 1e-6);
    const height = Math.max(maxY - minY, 1e-6);

    const padding = 18;
    const viewSize = 200;
    const scale = Math.min((viewSize - padding * 2) / width, (viewSize - padding * 2) / height);

    const toSvg = (point) => [
      (point[0] - minX) * scale + padding,
      (point[1] - minY) * scale + padding,
    ];

    const pointsSvg = points2d.map(toSvg);
    const edges = [
      [0, 1], [0, 2], [0, 3],
      [1, 4], [1, 5],
      [2, 4], [2, 6],
      [3, 5], [3, 6],
      [4, 7], [5, 7], [6, 7],
    ];

    return (
      <svg viewBox={`0 0 ${viewSize} ${viewSize}`} className="w-full h-56">
        <rect x="0" y="0" width={viewSize} height={viewSize} fill="transparent" stroke="#1f2937" />
        {edges.map(([start, end]) => (
          <line
            key={`${start}-${end}`}
            x1={pointsSvg[start][0]}
            y1={pointsSvg[start][1]}
            x2={pointsSvg[end][0]}
            y2={pointsSvg[end][1]}
            stroke="#34d399"
            strokeWidth="2"
            strokeLinecap="round"
          />
        ))}
      </svg>
    );
  };

  // 2. Handle file selection from the input
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setUploadMessage(''); 
    setTrajectoryFile(null); // Reset trajectory if main file changes
    setOutputFilePath(null); // Clear previous output
    setPreparedData(null);
    setSelectedPreparedId('');
    setPreparedName('');
    setPrepareProgressMessage('');
    setFrameSlice('');
    setSelectedKey('');
    setPrepareError('');
    setSelectedBatch(0);
    setActiveKeyDetails(null);
    setKeyDetailsError('');
    setLatticeDetails(null);
    setLatticeError('');
    setLatticeSaving(false);
    setLatticeMatrix([
      ['1', '0', '0'],
      ['0', '1', '0'],
      ['0', '0', '1'],
    ]);
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
    setUploadMessage('Uploading and preparing input...');
    setUploadMessageColor('text-blue-500');
    setOutputFilePath(null);
    setLatestResult('');
    setUploadProgress(0);
    setProgressPhase('upload');
    setPrepareError('');
    setPreparedData(null);
    setSelectedKey('');

    const formData = new FormData();
    formData.append('file', selectedFile);
    if (preparedName.trim()) {
      formData.append('name', preparedName.trim());
    }

    if (trajectoryFile) {
      formData.append('trajectory_file', trajectoryFile);
    }

    const targetUrl = `${API_URL}/prepare`;
    const allowedMimeTypes = ['.xyz', '.pdb', '.gro'];

    // Basic file type check before sending
    const fileExtension = selectedFile.name.split('.').pop().toLowerCase();
    if (!allowedMimeTypes.includes('.' + fileExtension)) { 
        setUploadMessage(`Error: File type .${fileExtension} is not supported.`);
        setUploadMessageColor('text-red-500');
        setIsUploading(false);
        setUploadProgress(0);
        setProgressPhase('idle');
        return;
    }

    const xhr = new XMLHttpRequest();
    xhr.open('POST', targetUrl, true);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = Math.round((event.loaded / event.total) * 100);
        setUploadProgress(percent);
      }
    };
    xhr.upload.onload = () => {
      setUploadProgress(100);
      setProgressPhase('prepare');
    };
    xhr.onload = () => {
      setProgressPhase('idle');
      try {
        const responseData = JSON.parse(xhr.responseText || '{}');
        if (xhr.status >= 200 && xhr.status < 300) {
          setUploadMessage('Upload complete. Preparing input...');
          setUploadMessageColor('text-blue-500');
          setProgressPhase('prepare');
          setUploadProgress(0);
          setLatticeError('');
          setLatticeSaving(false);
          if (responseData.job_id) {
            pollPrepareJob(responseData.job_id);
          } else {
            throw new Error('Backend did not return a prepare job id');
          }
        } else {
          throw new Error(responseData.detail || 'An unknown error occurred');
        }
      } catch (error) {
        console.error('Upload error:', error);
        setIsUploading(false);
        setProgressPhase('idle');
        setUploadMessage(`Error: ${error.message}`);
        setUploadMessageColor('text-red-500');
        setUploadProgress(0);
        setPrepareError('Unable to prepare input data. Please try again.');
      } finally {
        // Reset the file input visually
        if (document.getElementById('file-upload')) {
            document.getElementById('file-upload').value = '';
        }
        if (document.getElementById('trajectory-upload')) {
            document.getElementById('trajectory-upload').value = '';
        }
      }
    };
    xhr.onerror = () => {
      setIsUploading(false);
      setProgressPhase('idle');
      setUploadProgress(0);
      setUploadMessage('Error: Upload failed. Please try again.');
      setUploadMessageColor('text-red-500');
      setPrepareError('Upload failed before preparation could start.');
      if (document.getElementById('file-upload')) {
          document.getElementById('file-upload').value = '';
      }
      if (document.getElementById('trajectory-upload')) {
          document.getElementById('trajectory-upload').value = '';
      }
    };
    xhr.send(formData);
  };
  
  const handleTrajectoryFileChange = (event) => {
    setTrajectoryFile(event.target.files[0]);
  };

  const handleRunInference = () => {
    if (!preparedData?.id) {
      setUploadMessage('Prepare an input before running inference.');
      setUploadMessageColor('text-yellow-500');
      return;
    }
    if (!selectedModel) {
      setUploadMessage('Please select a model before running inference.');
      setUploadMessageColor('text-yellow-500');
      return;
    }

    setIsUploading(true);
    setUploadMessage('Running inference...');
    setUploadMessageColor('text-blue-500');
    setUploadProgress(0);
    setProgressPhase('inference');

    const formData = new FormData();
    formData.append('model_name', selectedModel);
    formData.append('destandardize', destandardize);
    if (frameSlice.trim()) {
      formData.append('frame_slice', frameSlice.trim());
    }

    fetch(`${API_URL}/infer/prepared/${preparedData.id}`, {
      method: 'POST',
      body: formData,
    })
      .then(response => {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
          return response.json().then(data => ({ ok: response.ok, data }));
        }
        return response.text().then(text => {
          throw new Error(text || 'Server returned non-JSON response');
        });
      })
      .then(({ ok, data }) => {
        if (ok) {
          setUploadMessage(`Success! ${data.atoms_predicted} atom predictions generated.`);
          setUploadMessageColor('text-green-500');
          const fullPath = data.output_file;
          const filename = fullPath.split('/').pop();
          setOutputFilePath(filename);
          setLatestResult(filename);
          fetchResults();
        } else {
          throw new Error(data.detail || 'An unknown error occurred');
        }
      })
      .catch(error => {
        console.error('Inference error:', error);
        setUploadMessage(`Error: ${error.message}`);
        setUploadMessageColor('text-red-500');
      })
      .finally(() => {
        setIsUploading(false);
        setProgressPhase('idle');
      });
  };

  const handleApplyLattice = () => {
    if (!preparedData?.id) {
      setLatticeError('Prepare an input before applying a lattice matrix.');
      return;
    }

    const parsedMatrix = latticeMatrix.map((row) => row.map((value) => parseFloat(value)));
    const matrixValid = parsedMatrix.every((row) => row.length === 3 && row.every((value) => Number.isFinite(value)));
    if (!matrixValid) {
      setLatticeError('Lattice matrix must contain numeric values for all 3x3 entries.');
      return;
    }

    setLatticeSaving(true);
    setLatticeError('');
    fetch(`${API_URL}/prepare/${preparedData.id}/lattice`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ matrix: parsedMatrix }),
    })
      .then(response => {
        if (!response.ok) throw new Error('Failed to apply lattice matrix');
        return response.json();
      })
      .then(data => {
        setPreparedData((prev) => (prev ? { ...prev, keys: data.keys || prev.keys } : prev));
        setSelectedKey('Lattice');
        fetchLatticeDetails(selectedBatch);
      })
      .catch(error => {
        console.error('Lattice apply error:', error);
        setLatticeError('Unable to apply lattice matrix.');
      })
      .finally(() => {
        setLatticeSaving(false);
      });
  };

  useEffect(() => {
    if (preparedData?.id && selectedKey) {
      fetchKeyDetails(selectedKey, selectedBatch);
    }
    if (preparedData?.id) {
      fetchLatticeDetails(selectedBatch);
    }
  }, [preparedData?.id, selectedKey, selectedBatch]);
  const downloadLink = outputFilePath ? `${API_URL}/download/${outputFilePath}` : '#';
  const buttonText = isUploading
    ? (progressPhase === 'prepare' ? 'Preparing...' : 'Uploading...')
    : 'Prepare Input';
  const allowedFileTypes = '.pdb, .gro, .xyz';
  const instructions = "Upload a PDB, GRO, or XYZ file. For PDB/GRO, predictions are in the B-factor column. For XYZ, they are in a new 'cs_iso' column. Trajectories are returned as a ZIP of PDBs or a multi-frame XYZ file.";
  const progressLabel = progressPhase === 'upload'
    ? `Uploading ${uploadProgress}%`
    : progressPhase === 'prepare'
      ? (prepareProgressMessage || 'Preparing input...')
      : progressPhase === 'inference'
        ? 'Running inference...'
        : '';

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center p-8 font-sans">
      <div className="w-full max-w-4xl space-y-8">
        <div className="space-y-4">
          <h1 className="text-4xl font-bold text-center text-white">GEqNMR Workflow</h1>
          <div className="flex items-center justify-center gap-3">
            <button
              type="button"
              onClick={() => setView('upload')}
              className={`px-4 py-2 rounded-md text-sm font-semibold border ${
                view === 'upload'
                  ? 'bg-gray-100 text-gray-900 border-gray-100'
                  : 'bg-gray-800 text-gray-200 border-gray-700 hover:bg-gray-700'
              }`}
            >
              Upload
            </button>
            <button
              type="button"
              onClick={() => setView('prepare')}
              disabled={!preparedData}
              className={`px-4 py-2 rounded-md text-sm font-semibold border ${
                view === 'prepare'
                  ? 'bg-gray-100 text-gray-900 border-gray-100'
                  : 'bg-gray-800 text-gray-200 border-gray-700 hover:bg-gray-700'
              } ${!preparedData ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              Prepared Input
            </button>
            <button
              type="button"
              onClick={openResults}
              className={`px-4 py-2 rounded-md text-sm font-semibold border ${
                view === 'results'
                  ? 'bg-gray-100 text-gray-900 border-gray-100'
                  : 'bg-gray-800 text-gray-200 border-gray-700 hover:bg-gray-700'
              }`}
            >
              Results
            </button>
            <button
              type="button"
              onClick={() => setView('visualize')}
              className={`px-4 py-2 rounded-md text-sm font-semibold border ${
                view === 'visualize'
                  ? 'bg-gray-100 text-gray-900 border-gray-100'
                  : 'bg-gray-800 text-gray-200 border-gray-700 hover:bg-gray-700'
              }`}
            >
              Visualize
            </button>
          </div>
        </div>

        {view === 'upload' && (
          <>
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

            <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-2xl font-semibold mb-4 text-gray-200">
                Run NMR Prediction
              </h2>
              <p className="text-gray-400 mb-4">{instructions}</p>
              
              <form onSubmit={handleUpload} className="space-y-4">
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

                <div>
                  <label htmlFor="prepared-name" className="block text-sm font-medium text-gray-400 mb-1">
                    Stored input name
                  </label>
                  <input
                    id="prepared-name"
                    type="text"
                    value={preparedName}
                    onChange={(event) => setPreparedName(event.target.value)}
                    placeholder={selectedFile?.name || 'Optional custom name'}
                    className="block w-full text-sm text-gray-200 bg-gray-700 border border-gray-600 rounded-md
                      py-2 px-3 focus:outline-none focus:ring-2 focus:ring-green-500"
                    disabled={isUploading}
                  />
                </div>

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

              {progressPhase !== 'idle' && (
                <div className="mt-5">
                  <div className="flex items-center justify-between text-xs uppercase tracking-wide text-gray-400 mb-2">
                    <span>{progressLabel}</span>
                    {(progressPhase === 'upload' || progressPhase === 'prepare') && <span>{uploadProgress}%</span>}
                  </div>
                  <div className="h-3 w-full rounded-full bg-gray-700 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        progressPhase === 'upload' ? 'bg-blue-500' : 'bg-green-500 animate-pulse'
                      }`}
                      style={{ width: progressPhase === 'inference' ? '100%' : `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {uploadMessage && (
                <div id="status-message" className={`mt-4 text-center font-medium ${uploadMessageColor}`}>
                  {uploadMessage}
                </div>
              )}
              {prepareError && (
                <div className="mt-2 text-center text-sm text-red-400">
                  {prepareError}
                </div>
              )}
            </div>
          </>
        )}

        {view === 'prepare' && (
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg space-y-6">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-2xl font-semibold text-gray-200">Prepared Inputs</h2>
                <p className="text-sm text-gray-400">
                  {preparedData ? `Selected: ${preparedData.name || preparedData.input_file}` : 'Select a stored input or upload a new one.'}
                </p>
                {preparedData?.num_molecules !== undefined && (
                  <p className="text-xs text-gray-500">
                    Structures detected: {preparedData.num_molecules}
                  </p>
                )}
              </div>
              <button
                type="button"
                onClick={() => setView('upload')}
                className="text-sm font-semibold px-4 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700"
              >
                Upload new input
              </button>
            </div>

            {prepareError && (
              <div className="text-sm text-red-400">{prepareError}</div>
            )}

            <div className="rounded-lg border border-gray-700 bg-gray-900/40 p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="text-xs uppercase tracking-wide text-gray-400">Stored inputs</div>
                <button
                  type="button"
                  onClick={fetchPreparedInputs}
                  className="text-xs text-green-300 hover:text-green-200"
                >
                  Refresh
                </button>
              </div>
              {preparedInputs.length === 0 ? (
                <p className="text-sm text-gray-500">No stored prepared inputs yet.</p>
              ) : (
                <div className="grid gap-3 md:grid-cols-2">
                  {preparedInputs.map((item) => {
                    const active = selectedPreparedId === item.id;
                    return (
                      <div
                        key={item.id}
                        className={`rounded-md border px-4 py-3 ${
                          active ? 'border-green-400 bg-gray-900/70' : 'border-gray-700 bg-gray-900/30'
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => openPreparedInput(item.id)}
                          className="block w-full text-left"
                        >
                          <div className="text-sm font-semibold text-gray-100">{item.name}</div>
                          <div className="text-xs text-gray-400">{item.input_file}</div>
                          <div className="text-xs text-gray-500">
                            {item.num_molecules} frame(s) · {formatTimestamp(item.modified)}
                          </div>
                        </button>
                        <button
                          type="button"
                          onClick={() => deletePreparedInput(item.id)}
                          className="mt-3 text-xs font-semibold text-red-300 hover:text-red-200"
                        >
                          Delete
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {preparedData && (
              <div className="space-y-6">
                <div className="rounded-lg border border-gray-700 bg-gray-900/40 p-4 space-y-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-100">Lattice &amp; PBC</h3>
                    <p className="text-xs text-gray-400">
                      {preparedData.keys?.some((key) => key.name === 'Lattice')
                        ? 'Lattice found in prepared input.'
                        : 'No lattice found. You can add one below.'}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-4 text-xs text-gray-300">
                    {(() => {
                      const pbcPresent = preparedData.keys?.find((key) => key.name === 'pbc_present');
                      const pbcDetected = preparedData.keys?.find((key) => key.name === 'pbc_detected');
                      const presentCount = pbcPresent?.counts?.true ?? 0;
                      const detectedCount = pbcDetected?.counts?.true ?? 0;
                      return (
                        <>
                          <span>Input lattice present: {presentCount > 0 ? 'Yes' : 'No'}</span>
                          <span>MDAnalysis detected: {detectedCount > 0 ? 'Yes' : 'No'}</span>
                        </>
                      );
                    })()}
                  </div>
                  <div className="grid gap-4 lg:grid-cols-2">
                    <div className="space-y-2">
                      <div className="grid grid-cols-3 gap-2">
                        {latticeMatrix.map((row, rowIndex) => (
                          row.map((value, colIndex) => (
                            <input
                              key={`lattice-${rowIndex}-${colIndex}`}
                              type="number"
                              value={value}
                              onChange={(event) => {
                                const nextMatrix = latticeMatrix.map((matrixRow, rowIdx) =>
                                  matrixRow.map((cellValue, colIdx) =>
                                    rowIdx === rowIndex && colIdx === colIndex ? event.target.value : cellValue
                                  )
                                );
                                setLatticeMatrix(nextMatrix);
                              }}
                              className="w-full rounded-md bg-gray-900 border border-gray-700 px-2 py-1 text-sm text-gray-200
                                focus:outline-none focus:ring-2 focus:ring-green-500"
                            />
                          ))
                        ))}
                      </div>
                      {latticeError && (
                        <p className="text-xs text-red-400">{latticeError}</p>
                      )}
                      <button
                        type="button"
                        onClick={handleApplyLattice}
                        disabled={latticeSaving || !preparedData}
                        className="inline-flex items-center justify-center bg-green-600 text-white text-sm font-semibold py-2 px-4 rounded-md
                          hover:bg-green-700 transition-colors duration-200 disabled:bg-gray-600 disabled:cursor-not-allowed"
                      >
                        {latticeSaving ? 'Applying...' : (preparedData.keys?.some((key) => key.name === 'Lattice') ? 'Update lattice matrix' : 'Apply lattice matrix')}
                      </button>
                    </div>
                    <div className="rounded-md border border-gray-700 bg-gray-900/60 p-3">
                      <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">3D Box Preview</div>
                      {renderLatticePreview(latticeMatrix)}
                    </div>
                  </div>
                </div>

                <div className="grid gap-6 lg:grid-cols-2">
                  <div className="space-y-3">
                    <div className="text-xs uppercase tracking-wide text-gray-400">Prepared Keys</div>
                    <div className="space-y-2">
                      {preparedData.keys.map((key) => {
                        const isActive = key.name === selectedKey;
                        return (
                          <button
                            key={key.name}
                            type="button"
                            onClick={() => setSelectedKey(key.name)}
                            className={`w-full text-left rounded-md border px-4 py-3 transition ${
                              isActive
                                ? 'border-green-400 bg-gray-900/50'
                                : 'border-gray-700 bg-gray-900/20 hover:bg-gray-900/40'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-semibold text-gray-100">{key.name}</span>
                              <span className="text-xs text-gray-400">{key.shape.join('x')}</span>
                            </div>
                            <div className="flex items-center justify-between text-xs text-gray-500">
                              <span>{key.dtype} · {key.size} values</span>
                              {key.name === 'Lattice' && (
                                <span className="text-xs text-green-300">3x3 matrix</span>
                              )}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                    {preparedData.num_molecules > 1 && (
                      <div className="mt-4">
                        <label htmlFor="batch-select" className="block text-xs uppercase tracking-wide text-gray-400 mb-2">
                          Batch index
                        </label>
                        <input
                          id="batch-select"
                          type="number"
                          min={0}
                          max={Math.max(preparedData.num_molecules - 1, 0)}
                          value={selectedBatch}
                          onChange={(event) => {
                            const maxBatch = Math.max(preparedData.num_molecules - 1, 0);
                            const nextValue = parseInt(event.target.value, 10);
                            const clampedValue = Number.isNaN(nextValue)
                              ? 0
                              : Math.min(Math.max(nextValue, 0), maxBatch);
                            setSelectedBatch(clampedValue);
                          }}
                          className="w-full rounded-md bg-gray-900 border border-gray-700 px-3 py-2 text-sm text-gray-200
                            focus:outline-none focus:ring-2 focus:ring-green-500"
                        />
                        <p className="text-xs text-gray-500 mt-1">
                          Available batches: 0 - {Math.max(preparedData.num_molecules - 1, 0)}
                        </p>
                      </div>
                    )}
                  </div>

                <div className="rounded-lg border border-gray-700 bg-gray-900/40 p-4">
                  {activeKey ? (
                    <div className="space-y-4">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-100">{activeKey.name}</h3>
                        <p className="text-xs text-gray-400">
                          Shape: {activeKey.shape.join('x')} · {activeKey.dtype}
                        </p>
                        {Number.isFinite(activeKey.batch_index) && (
                          <p className="text-xs text-gray-500">
                            Batch: {activeKey.batch_index}
                          </p>
                        )}
                      </div>

                      {keyDetailsError && (
                        <div className="text-xs text-red-400">{keyDetailsError}</div>
                      )}

                      {activeKey.name === 'Lattice' ? (
                        <div className="space-y-4">
                          <div className="grid grid-cols-3 gap-2 text-sm text-gray-200">
                            {latticeMatrix.map((row, rowIndex) => (
                              row.map((value, colIndex) => (
                                <div
                                  key={`lattice-detail-${rowIndex}-${colIndex}`}
                                  className="rounded-md bg-gray-900 border border-gray-700 px-2 py-1 text-center"
                                >
                                  {value}
                                </div>
                              ))
                            ))}
                          </div>
                          <div className="rounded-md border border-gray-700 bg-gray-900/60 p-3">
                            <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">3D Box Preview</div>
                            {renderLatticePreview(latticeMatrix)}
                          </div>
                        </div>
                      ) : (
                        <>
                          {activeKey.stats && (
                            <div className="grid grid-cols-2 gap-3 text-xs text-gray-300">
                              <div>Min: {activeKey.stats.min ?? '—'}</div>
                              <div>Max: {activeKey.stats.max ?? '—'}</div>
                              <div>Mean: {activeKey.stats.mean ?? '—'}</div>
                              <div>Std: {activeKey.stats.std ?? '—'}</div>
                            </div>
                          )}

                          {activeKey.counts && (
                            <div className="text-xs text-gray-300">
                              True: {activeKey.counts.true} · False: {activeKey.counts.false}
                            </div>
                          )}

                          {activeKey.sample && (
                            <div>
                              <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">Sample values</div>
                              <div className="text-xs text-gray-200 bg-gray-900 rounded-md p-3 overflow-auto max-h-32">
                                {JSON.stringify(activeKey.sample, null, 2)}
                              </div>
                            </div>
                          )}

                          <div>
                            <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">Distribution</div>
                            {renderHistogram(activeKey.histogram, Boolean(activeKey.counts))}
                          </div>
                        </>
                      )}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-400">Select a key to inspect its values.</p>
                  )}
                </div>
                </div>
              </div>
            )}

            <div className="border-t border-gray-700 pt-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-200">Run Inference</h3>
                {modelLoadError && (
                  <span className="text-xs text-red-400">{modelLoadError}</span>
                )}
              </div>
              <div className="grid gap-4 md:grid-cols-3">
                <div>
                  <label htmlFor="model-select-prepare" className="block text-sm font-medium text-gray-400 mb-1">
                    Model
                  </label>
                  <select
                    id="model-select-prepare"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    disabled={isUploading || modelOptions.length === 0}
                    className="block w-full text-sm text-gray-200 bg-gray-700 border border-gray-600 rounded-md
                      py-2 px-3 focus:outline-none focus:ring-2 focus:ring-green-500"
                  >
                    {modelOptions.map((modelName) => (
                      <option key={modelName} value={modelName}>
                        {modelName}
                      </option>
                    ))}
                  </select>
                  {!modelLoadError && modelOptions.length === 0 && (
                    <p className="text-xs text-yellow-400 mt-1">
                      No models found. Add .pth files to the backend models folder.
                    </p>
                  )}
                </div>

                <div>
                  <label htmlFor="frame-slice" className="block text-sm font-medium text-gray-400 mb-1">
                    Frame slice
                  </label>
                  <input
                    id="frame-slice"
                    type="text"
                    value={frameSlice}
                    onChange={(event) => setFrameSlice(event.target.value)}
                    placeholder="start:stop:step"
                    className="block w-full text-sm text-gray-200 bg-gray-700 border border-gray-600 rounded-md
                      py-2 px-3 focus:outline-none focus:ring-2 focus:ring-green-500"
                    disabled={isUploading}
                  />
                  <p className="text-xs text-gray-500 mt-1">Leave empty to infer all frames.</p>
                </div>

                <div className="flex items-center justify-center md:justify-start pt-6">
                  <input
                    id="destandardize-checkbox-prepare"
                    type="checkbox"
                    checked={destandardize}
                    onChange={(e) => setDestandardize(e.target.checked)}
                    className="w-4 h-4 text-indigo-600 bg-gray-700 border-gray-600 rounded focus:ring-indigo-500"
                  />
                  <label htmlFor="destandardize-checkbox-prepare" className="ml-2 text-sm font-medium text-gray-300">
                    De-standardize predictions before saving
                  </label>
                </div>
              </div>

              <button
                type="button"
                onClick={handleRunInference}
                disabled={isUploading || !preparedData || !selectedModel}
                className="w-full bg-green-600 text-white font-bold py-3 px-4 rounded-md
                  hover:bg-green-700
                  focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50
                  disabled:bg-gray-600 disabled:cursor-not-allowed"
              >
                {isUploading && progressPhase === 'inference' ? 'Running Inference...' : 'Run Inference'}
              </button>

              {progressPhase === 'inference' && (
                <div>
                  <div className="flex items-center justify-between text-xs uppercase tracking-wide text-gray-400 mb-2">
                    <span>{progressLabel}</span>
                  </div>
                  <div className="h-3 w-full rounded-full bg-gray-700 overflow-hidden">
                    <div className="h-full rounded-full bg-green-500 animate-pulse" style={{ width: '100%' }} />
                  </div>
                </div>
              )}

              {uploadMessage && (
                <div className={`text-center font-medium ${uploadMessageColor}`}>
                  {uploadMessage}
                </div>
              )}

              {outputFilePath && (
                <div className="mt-2 text-center space-y-4">
                  <a
                    href={downloadLink}
                    download={outputFilePath}
                    className="inline-flex items-center justify-center bg-blue-600 text-white font-bold py-2 px-6 rounded-md
                      hover:bg-blue-700 transition-colors duration-200"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                    Download Output: {outputFilePath}
                  </a>
                  <p className="text-sm text-gray-500">
                    ({outputFilePath.endsWith('.zip') ? 'ZIP archive of PDB frames.' : (outputFilePath.endsWith('.xyz') ? "Extended XYZ file with 'cs_iso' column." : 'Predictions are in the B-factor column.')})
                  </p>
                  <button
                    type="button"
                    onClick={goToResults}
                    className="inline-flex items-center justify-center bg-gray-100 text-gray-900 font-semibold py-2 px-6 rounded-md
                      hover:bg-white transition-colors duration-200"
                  >
                    Go to results
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {view === 'results' && (
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-2xl font-semibold text-gray-200">Results</h2>
                <p className="text-sm text-gray-400">
                  Download any predicted structure. Latest runs appear first.
                </p>
              </div>
              <button
                type="button"
                onClick={fetchResults}
                className="text-sm font-semibold px-4 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700"
                disabled={isLoadingResults}
              >
                {isLoadingResults ? 'Loading...' : 'Refresh'}
              </button>
            </div>

            {resultsError && (
              <div className="text-sm text-red-400 mb-4">{resultsError}</div>
            )}

            {!resultsError && results.length === 0 && !isLoadingResults && (
              <div className="text-sm text-gray-400">
                No results yet. Run an inference to generate output files.
              </div>
            )}

            <div className="space-y-3">
              {results.map((result) => {
                const isHighlighted = highlightedResult === result.name;
                return (
                  <div
                    key={result.name}
                    className={`flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between rounded-lg border px-4 py-3 ${
                      isHighlighted
                        ? 'border-green-400 bg-gray-900/50'
                        : 'border-gray-700 bg-gray-900/20'
                    }`}
                  >
                    <div>
                      <div className="text-sm font-semibold text-gray-100">{result.name}</div>
                      <div className="text-xs text-gray-400">
                        {formatBytes(result.size_bytes)} · {formatTimestamp(result.modified)}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {isHighlighted && (
                        <span className="text-xs uppercase tracking-wide text-green-300 border border-green-400/60 px-2 py-1 rounded-full">
                          Latest
                        </span>
                      )}
                      <a
                        href={`${API_URL}/download/${result.name}`}
                        download={result.name}
                        className="inline-flex items-center justify-center bg-blue-600 text-white text-sm font-semibold py-2 px-4 rounded-md
                          hover:bg-blue-700 transition-colors duration-200"
                      >
                        Download
                      </a>
                      <button
                        type="button"
                        onClick={() => {
                          fetch(`${API_URL}/results/${result.name}`, { method: 'DELETE' })
                            .then(response => {
                              if (!response.ok) throw new Error('Failed to delete result');
                              if (highlightedResult === result.name) {
                                setHighlightedResult('');
                              }
                              if (latestResult === result.name) {
                                setLatestResult('');
                                setOutputFilePath(null);
                              }
                              fetchResults();
                            })
                            .catch(error => {
                              console.error('Delete error:', error);
                              setResultsError('Unable to delete the selected result.');
                            });
                        }}
                        className="inline-flex items-center justify-center border border-red-400/60 text-red-200 text-sm font-semibold py-2 px-4 rounded-md
                          hover:bg-red-500/10 transition-colors duration-200"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {view === 'visualize' && (
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg space-y-6">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <h2 className="text-2xl font-semibold text-gray-200">Prediction Visualization</h2>
                <p className="text-sm text-gray-400">
                  Compare atom-wise predicted chemical shifts across saved inference results.
                </p>
              </div>
              <button
                type="button"
                onClick={fetchPredictionResults}
                className="text-sm font-semibold px-4 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700"
                disabled={isLoadingPredictionResults}
              >
                {isLoadingPredictionResults ? 'Loading...' : 'Refresh'}
              </button>
            </div>

            {predictionResultsError && (
              <div className="text-sm text-red-400">{predictionResultsError}</div>
            )}

            <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
              <div className="space-y-5">
                <div className="rounded-lg border border-gray-700 bg-gray-900/40 p-4">
                  <div className="text-xs uppercase tracking-wide text-gray-400 mb-3">Results</div>
                  {predictionResults.length === 0 && !isLoadingPredictionResults ? (
                    <p className="text-sm text-gray-500">No prediction metadata found yet.</p>
                  ) : (
                    <div className="max-h-64 overflow-auto space-y-2">
                      {predictionResults.map((result) => {
                        const checked = selectedPredictionResults.includes(result.name);
                        return (
                          <label key={result.name} className="flex items-start gap-2 rounded-md border border-gray-700 bg-gray-900/30 px-3 py-2">
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={(event) => {
                                if (event.target.checked) {
                                  setSelectedPredictionResults((prev) => [...prev, result.name]);
                                } else {
                                  setSelectedPredictionResults((prev) => prev.filter((name) => name !== result.name));
                                }
                              }}
                              className="mt-1 h-4 w-4 rounded border-gray-600 bg-gray-700 text-green-500 focus:ring-green-500"
                            />
                            <span>
                              <span className="block text-sm text-gray-100 break-all">{result.name}</span>
                              <span className="block text-xs text-gray-500">{formatTimestamp(result.modified)}</span>
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  )}
                </div>

                <div className="rounded-lg border border-gray-700 bg-gray-900/40 p-4">
                  <div className="text-xs uppercase tracking-wide text-gray-400 mb-3">Mode</div>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => setVisualizationMode('violin')}
                      className={`rounded-md px-3 py-2 text-sm font-semibold border ${
                        visualizationMode === 'violin'
                          ? 'bg-green-500 text-gray-950 border-green-400'
                          : 'bg-gray-900 text-gray-200 border-gray-700 hover:bg-gray-700'
                      }`}
                    >
                      Violin
                    </button>
                    <button
                      type="button"
                      onClick={() => setVisualizationMode('pair')}
                      className={`rounded-md px-3 py-2 text-sm font-semibold border ${
                        visualizationMode === 'pair'
                          ? 'bg-green-500 text-gray-950 border-green-400'
                          : 'bg-gray-900 text-gray-200 border-gray-700 hover:bg-gray-700'
                      }`}
                    >
                      2D
                    </button>
                  </div>
                </div>

                <div className="rounded-lg border border-gray-700 bg-gray-900/40 p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="text-xs uppercase tracking-wide text-gray-400">Atoms</div>
                    {visualizationMode === 'violin' && atomOptions.length > 0 && (
                      <button
                        type="button"
                        onClick={() => setSelectedAtomIds(atomOptions.map((atom) => atom.id))}
                        className="text-xs text-green-300 hover:text-green-200"
                      >
                        Select all
                      </button>
                    )}
                  </div>

                  {atomOptions.length === 0 ? (
                    <p className="text-sm text-gray-500">Select result files to load atom labels.</p>
                  ) : visualizationMode === 'violin' ? (
                    <div className="max-h-80 overflow-auto space-y-2">
                      {atomOptions.map((atom) => (
                        <label key={atom.id} className="flex items-center gap-2 text-sm text-gray-200">
                          <input
                            type="checkbox"
                            checked={selectedAtomIds.includes(atom.id)}
                            onChange={(event) => {
                              if (event.target.checked) {
                                setSelectedAtomIds((prev) => [...prev, atom.id]);
                              } else {
                                setSelectedAtomIds((prev) => prev.filter((id) => id !== atom.id));
                              }
                            }}
                            className="h-4 w-4 rounded border-gray-600 bg-gray-700 text-green-500 focus:ring-green-500"
                          />
                          <span>{atom.label}</span>
                        </label>
                      ))}
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {atomOptions.map((atom) => {
                        const checked = selectedPairAtomIds.includes(atom.id);
                        return (
                          <label key={atom.id} className="flex items-center gap-2 text-sm text-gray-200">
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={(event) => {
                                if (event.target.checked) {
                                  setSelectedPairAtomIds((prev) => [...prev, atom.id].slice(-2));
                                } else {
                                  setSelectedPairAtomIds((prev) => prev.filter((id) => id !== atom.id));
                                }
                              }}
                              className="h-4 w-4 rounded border-gray-600 bg-gray-700 text-green-500 focus:ring-green-500"
                            />
                            <span>{atom.label}</span>
                          </label>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded-lg border border-gray-700 bg-gray-900/40 p-4 min-h-[520px]">
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-gray-100">
                    {visualizationMode === 'violin' ? 'Atom Distributions' : 'Pair Distribution'}
                  </h3>
                  <p className="text-xs text-gray-500">
                    {visualizationMode === 'violin'
                      ? 'Each atom group contains one colored violin per selected result.'
                      : 'Each point pairs frame-matched predictions for the two selected atoms.'}
                  </p>
                </div>
                {visualizationMode === 'violin' ? renderViolinPlot() : renderPairPlot()}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
