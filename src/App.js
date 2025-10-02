// src/App.js
import React, { useState } from 'react';
import './App.css';
import RaceEventSelector from './components/RaceEventSelector';
import TrackSimulation from './components/TrackSimulation';
import DriverPosition from './components/DriverPosition';

function App() {
  const [year, setYear] = useState('');
  const [eventName, setEventName] = useState('');

  return (
    <div className="App">
      <h1>Formula 1 Race Simulation</h1>
      <RaceEventSelector setYear={setYear} setEventName={setEventName} />
      {year && eventName && (
        <>
          <TrackSimulation year={year} eventName={eventName} />
          <DriverPosition year={year} eventName={eventName} />
        </>
      )}
    </div>
  );
}

export default App;
