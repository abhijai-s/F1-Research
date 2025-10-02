import React, { useState, useEffect } from 'react';
import axios from 'axios';

const RaceEventSelector = ({ setYear, setEventName }) => {
  const [year, setLocalYear] = useState('');
  const [events, setEvents] = useState([]);
  const [selectedEvent, setSelectedEvent] = useState('');

  // Fetch events based on selected year
  useEffect(() => {
    const fetchEvents = async () => {
      if (year) {
        try {
          const response = await axios.get(`http://127.0.0.1:8000/events/${year}`);
          console.log("Event data received:", response.data.events);  // Add this line to log event data
          setEvents(response.data.events);
        } catch (error) {
          console.error("Error fetching events:", error);
        }
      }
    };

    fetchEvents();
  }, [year]);

  const handleYearChange = (e) => {
    const selectedYear = e.target.value;
    setLocalYear(selectedYear);
    setYear(selectedYear); // This will be used by the parent component
  };

  const handleEventChange = (e) => {
    const selectedEventName = e.target.value;
    setSelectedEvent(selectedEventName);
    setEventName(selectedEventName); // This will be used by the parent component
  };

  return (
    <div>
      <h2>Select a Year and Event</h2>
      <select value={year} onChange={handleYearChange}>
        <option value="">Select Year</option>
        <option value="2020">2020</option>
        <option value="2021">2021</option>
        <option value="2022">2022</option>
        <option value="2023">2023</option>
        <option value="2024">2024</option>
        <option value="2025">2025</option>
      </select>

      <select value={selectedEvent} onChange={handleEventChange}>
        <option value="">Select Event</option>
        {events.map((event, index) => (
          <option key={index} value={event.EventName}>
            {event.EventName}
          </option>
        ))}
      </select>
    </div>
  );
};

export default RaceEventSelector;
