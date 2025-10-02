// src/components/DriverPosition.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';

const DriverPosition = ({ year, eventName }) => {
  const [driverPositions, setDriverPositions] = useState({});
  const [drivers, setDrivers] = useState({}); // To store driver abbreviations
  const [loading, setLoading] = useState(true); // To handle loading state
  const [error, setError] = useState(null); // To handle errors

  // Fetch driver position data
  useEffect(() => {
    const fetchDriverPositions = async () => {
      try {
        const response = await axios.get(`http://127.0.0.1:8000/race/${year}/${eventName}/positions`);
        console.log("API response:", response.data); // Debugging
        // Ensure response.data.positions exists
        if (response.data && response.data.positions) {
          setDriverPositions(response.data.positions);
          setDrivers(response.data.drivers || {});
        } else {
          throw new Error("Positions data is missing in the response.");
        }
        setLoading(false);
      } catch (error) {
        console.error("Error fetching driver positions:", error);
        setError(error.message);
        setLoading(false);
      }
    };

    if (year && eventName) {
      fetchDriverPositions();
    }
  }, [year, eventName]);

  // Render driver positions
  const renderDriverPositions = () => {
    // Safeguard against undefined or null driverPositions
    if (!driverPositions || typeof driverPositions !== 'object') {
      return <text x="10" y="20" fill="red">No driver positions available.</text>;
    }

    return Object.entries(driverPositions).map(([driverNum, positions]) => {
      if (!positions || !Array.isArray(positions) || positions.length === 0) {
        return null; // Skip if no positions
      }

      const lastPosition = positions[positions.length - 1];
      if (!lastPosition || typeof lastPosition.X !== 'number' || typeof lastPosition.Y !== 'number') {
        return null; // Skip if position data is incomplete
      }

      return (
        <g key={driverNum}>
          <circle
            cx={lastPosition.X}
            cy={lastPosition.Y}
            r={5}
            fill="red"
            stroke="black"
          />
          {/* Display driver abbreviation if available */}
          {drivers[driverNum] && (
            <text
              x={lastPosition.X + 8} // Offset to prevent overlap with circle
              y={lastPosition.Y + 3}
              fontSize="10"
              fill="black"
            >
              {drivers[driverNum]}
            </text>
          )}
        </g>
      );
    });
  };

  if (loading) {
    return <div>Loading driver positions...</div>;
  }

  if (error) {
    return <div style={{ color: 'red' }}>Error: {error}</div>;
  }

  
};

export default DriverPosition;
