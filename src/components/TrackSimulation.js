import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './TrackSimulation.css';

const TrackSimulation = ({ year, eventName }) => {
  const canvasRef = useRef(null);
  const [trackData, setTrackData] = useState([]);
  const [driverPositions, setDriverPositions] = useState({});
  const [driverAbbreviations, setDriverAbbreviations] = useState({});
  const [currentIndex, setCurrentIndex] = useState(0);
  const [maxFrames, setMaxFrames] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [sliderValue, setSliderValue] = useState(0);
  const [speed, setSpeed] = useState(200);
  const [pitstopData, setPitstopData] = useState(null);
  const [uniqueDrivers, setUniqueDrivers] = useState([]);
  const [accuracyScore, setAccuracyScore] = useState(null);
  const [classificationReport, setClassificationReport] = useState(null);
  const [confusionMatrix, setConfusionMatrix] = useState(null);
  const [showGraph, setShowGraph] = useState(false);

  const animationRef = useRef(null);

  async function fetchPitstopData(year, eventName) {
    try {
      const response = await axios.get(`http://localhost:8000/race/${year}/${eventName}/pitstops`);
      const data = response.data;
      if (!data.error) {
        setPitstopData(data);
        setUniqueDrivers(data.unique_drivers || []);
      } else {
        console.error(data.error);
      }
    } catch (error) {
      console.error("Error fetching pit stop data:", error);
    }
  }

  async function fetchPerformanceReport(year, eventName) {
    try {
      const response = await axios.post("http://localhost:8000/predict-pitstop", {
        year,
        event_name: eventName
      });
      const data = response.data;
      if (data.accuracy !== null) {
        setAccuracyScore(data.accuracy);
        setClassificationReport(data.classification_report);
        setConfusionMatrix(data.confusion_matrix);
      } else {
        console.error("Error: Accuracy score not returned from API.");
      }
    } catch (error) {
      console.error("Error fetching performance report:", error);
    }
  }

  useEffect(() => {
    if (year && eventName) {
      fetchPitstopData(year, eventName);
      fetchPerformanceReport(year, eventName);
    }
  }, [year, eventName]);

  useEffect(() => {
    const fetchTrackData = async () => {
      try {
        const response = await axios.get(`http://127.0.0.1:8000/track/${year}/${eventName}`);
        setTrackData(response.data.track || []);
      } catch (error) {
        console.error('Error fetching track data:', error);
      }
    };
    if (year && eventName) {
      fetchTrackData();
    }
  }, [year, eventName]);

  useEffect(() => {
    const fetchDriverData = async () => {
      try {
        const positionResponse = await axios.get(`http://127.0.0.1:8000/race/${year}/${eventName}/positions`);
        setDriverPositions(positionResponse.data.positions || {});

        const abbreviationResponse = await axios.get(`http://127.0.0.1:8000/race/${year}/${eventName}/drivers`);
        setDriverAbbreviations(abbreviationResponse.data.drivers || {});

        const maxFrames = Object.values(positionResponse.data.positions || {})[0]?.length || 0;
        setMaxFrames(maxFrames);
      } catch (error) {
        console.error('Error fetching driver data:', error);
      }
    };
    if (year && eventName) {
      fetchDriverData();
    }
  }, [year, eventName]);

  useEffect(() => {
    if (maxFrames > 0 && Object.keys(driverPositions).length > 0 && isPlaying) {
      startAnimation();
    }
    return () => stopAnimation();
  }, [maxFrames, driverPositions, isPlaying, speed]);

  const startAnimation = () => {
    let lastTime = 0;
    const animate = (currentTime) => {
      if (isPlaying) {
        const delta = currentTime - lastTime;
        if (delta > speed) {
          setCurrentIndex((prevIndex) => {
            const newIndex = prevIndex + 1;
            if (newIndex >= maxFrames) {
              setIsPlaying(false);
              setShowGraph(true);
              return maxFrames - 1;
            }
            return newIndex;
          });
          lastTime = currentTime;
        }
        animationRef.current = requestAnimationFrame(animate);
      }
    };
    animationRef.current = requestAnimationFrame(animate);
  };

  const stopAnimation = () => {
    cancelAnimationFrame(animationRef.current);
  };

  const drawTrack = (ctx) => {
    if (!trackData.length) return;
    const padding = 100;
    const { minX, maxX, minY, maxY } = getBoundingBox(trackData);
    const width = maxX - minX;
    const height = maxY - minY;
    const scaleX = (ctx.canvas.width - 2 * padding) / width;
    const scaleY = (ctx.canvas.height - 2 * padding) / height;

    ctx.beginPath();
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 4;
    trackData.forEach((point, index) => {
      const x = (point.X - minX) * scaleX + padding;
      const y = (point.Y - minY) * scaleY + padding;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  const drawDrivers = (ctx) => {
    if (!Object.keys(driverPositions).length) return;
    const padding = 100;
    const { minX, maxX, minY, maxY } = getBoundingBox(trackData);
    const width = maxX - minX;
    const height = maxY - minY;
    const scaleX = (ctx.canvas.width - 2 * padding) / width;
    const scaleY = (ctx.canvas.height - 2 * padding) / height;

    const colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta'];
    let colorIndex = 0;

    Object.entries(driverPositions).forEach(([driverNum, positions]) => {
      if (positions.length > currentIndex) {
        const { X, Y } = positions[currentIndex];
        const x = (X - minX) * scaleX + padding;
        const y = (Y - minY) * scaleY + padding;

        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = colors[colorIndex % colors.length];
        ctx.fill();

        const driverAbbr = driverAbbreviations[driverNum] || driverNum;
        ctx.font = '12px Arial';
        ctx.fillStyle = 'white';
        ctx.fillText(driverAbbr, x + 10, y - 10);
      }
      colorIndex++;
    });
  };

  const getBoundingBox = (data) => {
    const xValues = data.map((point) => point.X);
    const yValues = data.map((point) => point.Y);
    return { minX: Math.min(...xValues), maxX: Math.max(...xValues), minY: Math.min(...yValues), maxY: Math.max(...yValues) };
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawTrack(ctx);
    drawDrivers(ctx);
  }, [trackData, driverPositions, driverAbbreviations, currentIndex]);

  const plotData = pitstopData
    ? ["predicted_only", "actual_only", "both"].map(category => ({
        x: pitstopData[category]?.map(item => item.lap) || [],
        y: pitstopData[category]?.map(item => item.driver) || [],
        mode: 'markers',
        type: 'scatter',
        name: category === "predicted_only" 
          ? "Predicted but didn't pit" 
          : category === "actual_only" 
          ? "Pit but not predicted" 
          : "Pit and predicted",
        marker: { color: category === "predicted_only" ? 'blue' : category === "actual_only" ? 'red' : '#e6760eff', size: 12 },
      }))
    : [];

  const renderClassificationReport = () => {
    if (!classificationReport) return null;
    const headers = ["Label", "Precision", "Recall", "F1-Score", "Support"];
    const rows = Object.keys(classificationReport).map((label) => {
      const { precision, recall, "f1-score": f1Score, support } = classificationReport[label];
      return (
        <tr key={label}>
          <td>{label}</td>
          <td>{(precision * 100).toFixed(2)}%</td>
          <td>{(recall * 100).toFixed(2)}%</td>
          <td>{(f1Score * 100).toFixed(2)}%</td>
          <td>{support}</td>
        </tr>
      );
    });
    return (
      <table style={{ width: "100%", color: "white", textAlign: "center" }}>
        <thead>
          <tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    );
  };

  const renderConfusionMatrix = () => {
    if (!confusionMatrix) return null;
    return (
      <table style={{ width: "100%", color: "white", textAlign: "center", marginTop: "20px" }}>
        <thead>
          <tr>
            <th></th>
            {confusionMatrix[0].map((_, index) => <th key={index}>Pred {index}</th>)}
          </tr>
        </thead>
        <tbody>
          {confusionMatrix.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <th>Actual {rowIndex}</th>
              {row.map((value, colIndex) => <td key={colIndex}>{value}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  return (
    <div>
      <h2>Track Simulation</h2>
      <canvas ref={canvasRef} width="800" height="600" style={{ border: '1px solid black' }}></canvas>
      <div>
        <button onClick={() => setIsPlaying(!isPlaying)}>{isPlaying ? "Pause" : "Play"}</button>
        <label>
          Seek:
          <input
            type="range"
            min="0"
            max={maxFrames - 1}
            value={sliderValue}
            onChange={(event) => {
              const seekValue = parseInt(event.target.value, 10);
              setCurrentIndex(seekValue);
              setSliderValue(seekValue);
            }}
          />
        </label>
        <label>
          Speed:
          <input type="range" min="200" max="250" value={speed} onChange={(e) => setSpeed(parseInt(e.target.value, 10))} />
          {speed}ms per frame
        </label>
      </div>
      {showGraph && (
        <>
          <h2 style={{ color: 'red', textAlign: 'center' }}>Performance Report</h2>
          {renderClassificationReport()}
          <h3 style={{ color: 'red', textAlign: 'center', marginTop: "20px" }}>Confusion Matrix</h3>
          {renderConfusionMatrix()}
          <h2 style={{ color: 'red', textAlign: 'center', marginTop: "20px" }}>Predicted vs Actual Pit Stops</h2>
          <Plot
            data={plotData}
            layout={{
              title: {
                  text: 'Predicted vs Actual Pit Stops',
                  font: {
                    size: 28,
                    color: 'black',
                    family: 'Arial, sans-serif, bold' // <-- Made main title bold
                  }},
              xaxis: { title: 'Lap', 
                color: 'black',
                titlefont: {
                    size: 26, // <-- SETS title size
                    family: 'Arial, sans-serif, bold' // <-- MAKES title bold
                },
                gridcolor: '#949494ff',
                 tickfont: {
                  size: 16 ,
                  family: 'Arial, sans-serif, bold'
                }
              
              },
              yaxis: {
                title: 'Driver',
                type: 'category',
                categoryarray: uniqueDrivers,
                automargin: true,
                color: 'black',
                titlefont: {
                    size: 26, // <-- SETS title size
                    family: 'Arial, sans-serif, bold' // <-- MAKES title bold
                },
                gridcolor: '#949494ff',
                tickfont: {
                  size: 13.5 ,
                  family: 'Arial, sans-serif, bold'
                }
                
              },
              plot_bgcolor: "#f9f9f9",
              paper_bgcolor: "#f9f9f9",
              font: { color: "black" },
              margin: { l: 150, r: 50, t: 50, b: 140 },
              showlegend: true,
              legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.35,yanchor: 'top', 
                font: { // <-- ADD THIS OBJECT
                size: 25,
                family: 'Arial, sans-serif, bold'
              } },
              hovermode: 'closest',
            }}
            config={{ responsive: true }}
            style={{ width: "100%", height: "600px" }}
          />
        </>
      )}
    </div>
  );
};

export default TrackSimulation;