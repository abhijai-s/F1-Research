from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import fastf1
import numpy as np
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score
from fastapi.responses import JSONResponse
import numpy as np
from preprocessing import preprocess_lap_data  # Assuming this is your custom preprocessing module
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionInput(BaseModel):
    year: int
    event_name: str


# Load the trained model
# model = tf.keras.models.load_model("models/lstm_model2.keras")
model = load_model("models/sofar_best_lstm_model_.h5")
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to your frontend's address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-pitstop")
async def predict_pitstop(input: PredictionInput):
    session = fastf1.get_session(input.year, input.event_name, 'R')
    session.load()
    laps = session.laps

    # Preprocess lap data for model prediction
    preprocessed_data, y_actual = preprocess_lap_data(laps, input.event_name)

    if preprocessed_data.ndim == 3 and preprocessed_data.shape[1:] == model.input_shape[1:]:
        y_pred_probs = model.predict(preprocessed_data)
    else:
        raise ValueError("Model input shape mismatch")

    y_pred = (y_pred_probs > 0.5).astype(int).flatten().tolist()

    if len(y_actual) == len(y_pred):
        accuracy = accuracy_score(y_actual, y_pred) * 100
        # Generate classification report and confusion matrix
        classification_rep = classification_report(y_actual, y_pred, output_dict=True)
        confusion_mat = confusion_matrix(y_actual, y_pred).tolist()  # Convert to list for JSON serialization
    else:
        accuracy = None
        classification_rep = {}
        confusion_mat = []

    logger.info("Model loaded successfully")
    model.summary()


    return {
        "predicted": y_pred,
        # "actual": y_actual.tolist(),
        "accuracy": accuracy,
        "classification_report": classification_rep,
        "confusion_matrix": confusion_mat
    }

@app.get("/race/{year}/{event_name}/pitstops")
async def get_pitstop_comparison(year: int, event_name: str):
    try:
        print(f"Fetching pit stop data for {year}, {event_name}")

        # Enable caching and fetch session
        fastf1.Cache.enable_cache("cache\F1 cache")
        session = fastf1.get_session(year, event_name, 'R')
        session.load()
        print("Session loaded successfully.")

        # Check if laps data is loaded
        laps = session.laps
        if laps.empty:
            print("No lap data found.")
            return {"error": "No lap data available"}, 500
        
        print(f"Laps loaded: {len(laps)}")

        # Get unique drivers in the session
        unique_drivers = laps["Driver"].unique().tolist()

        # Step 1: Preprocess lap data
        X_scaled, y_actual = preprocess_lap_data(laps, event_name)
        print(f"Preprocessed data shape: {X_scaled.shape}")

        # Step 2: Make predictions
        y_pred_probs = model.predict(X_scaled).flatten()
        y_pred = (y_pred_probs > 0.5).astype(int).tolist()
        print(f"Predictions (first 10): {y_pred[:10]}")

        # Step 3: Prepare structured data for frontend
        drivers = laps["Driver"].tolist()
        lap_numbers = laps["LapNumber"].tolist()

        predicted_only = []
        actual_only = []
        both = []

        for i in range(len(y_pred)):
            if y_pred[i] == 1 and y_actual[i] == 1:
                both.append({"lap": lap_numbers[i], "driver": drivers[i]})
            elif y_pred[i] == 1 and y_actual[i] == 0:
                predicted_only.append({"lap": lap_numbers[i], "driver": drivers[i]})
            elif y_pred[i] == 0 and y_actual[i] == 1:
                actual_only.append({"lap": lap_numbers[i], "driver": drivers[i]})

        return {
            "predicted_only": predicted_only,
            "actual_only": actual_only,
            "both": both,
            "unique_drivers": unique_drivers  # Add unique drivers to the response
        }
    
    except Exception as e:
        print(f"Error fetching pit stop comparison: {str(e)}")
        return {"error": "Failed to load pit stop data"}, 500


    
    
from fastapi.responses import JSONResponse
import numpy as np

@app.get("/race/{year}/{event_name}/raw_laps")
async def get_raw_lap_data(year: int, event_name: str):
    try:
        print(f"Fetching raw lap data for {year}, {event_name}")

        # Enable FastF1 cache
        fastf1.Cache.enable_cache("cache\F1 cache")
        
        # Load the race session
        session = fastf1.get_session(year, event_name, 'R')
        session.load()
        print("Session loaded successfully.")
        
        # Get lap data
        laps = session.laps
        if laps.empty:
            print("No lap data found.")
            return JSONResponse(content={"error": "No lap data available"}, status_code=500)

        # Convert relevant columns to JSON-serializable format
        laps = laps[['Time', 'Team', 'Driver', 'DriverNumber', 'Position', 'LapTime', 'LapNumber', 
                     'Stint', 'PitOutTime', 'PitInTime', 'TrackStatus', 'Compound', 'TyreLife']].copy()

        # Convert timedelta columns to seconds
        timedelta_columns = ['Time', 'LapTime', 'PitOutTime', 'PitInTime']
        for col in timedelta_columns:
            if col in laps.columns:
                laps[col] = laps[col].apply(lambda x: x.total_seconds() if pd.notnull(x) else None)

        # Handle TrackStatus NaN values
        laps['TrackStatus'] = laps['TrackStatus'].replace({np.nan: None})
        
        # Convert data types as specified
        laps['DriverNumber'] = laps['DriverNumber'].fillna(0).astype(int)
        laps['LapNumber'] = laps['LapNumber'].fillna(0).astype(int)
        laps['Stint'] = laps['Stint'].fillna(0).astype(int)
        laps['TyreLife'] = laps['TyreLife'].fillna(0).astype(int)
        
        # Handle NaN in Position
        laps['Position'] = laps['Position'].replace({np.nan: None})
        
        # Replace infinities or NaNs with None for JSON serialization
        laps.replace([np.inf, -np.inf, np.nan], None, inplace=True)

        # Convert the DataFrame to a dictionary format suitable for JSON response
        laps_data = laps.head(10).to_dict(orient="records")  # Limit to 10 records for testing

        # Return data as JSON response
        return JSONResponse(content={"laps": laps_data},status_code=200)
    
        print("Final laps data for JSON response:", laps_data)

    except Exception as e:
        print(f"Error fetching raw lap data: {str(e)}")
        return JSONResponse(content={"error": "Failed to load raw lap data"}, status_code=500)




# @app.get("/race/{year}/{event_name}/pitstops")
# async def get_pitstop_comparison(year: int, event_name: str):
#     try:
#         fastf1.Cache.enable_cache("C:/Users/acer/F1 cache")
        
#         # Step 1: Load the race session
#         session = fastf1.get_session(year, event_name, 'R')
#         session.load()
#         laps = session.laps

#         # Step 2: Preprocess lap data to get actual `HasPitstop`
#         preprocessed_data = preprocess_lap_data(laps)

#         # Extract actual pit stops from the preprocessed data
#         actual_pitstops = {}
#         for driver_id in preprocessed_data['Driver'].unique():
#             driver_laps = preprocessed_data[preprocessed_data['Driver'] == driver_id]
#             actual_pits = driver_laps[driver_laps['HasPitStop'] == 1]['LapNumber'].tolist()
#             actual_pitstops[driver_id] = actual_pits

#         # Step 3: Run predictions to determine predicted pit stops
#         # Make sure to drop the target 'HasPitStop' from features when predicting
#         X = preprocessed_data.drop(columns=['HasPitStop'])
#         predicted_probs = model.predict(X)
#         predicted_pitstops = (predicted_probs > 0.5).astype(int).flatten()  # Threshold predictions at 0.5

#         # Step 4: Categorize each lap for each driver
#         categorized_pitstops = {driver_id: {"predicted_only": [], "actual_only": [], "both": []} 
#                                 for driver_id in preprocessed_data['Driver'].unique()}

#         for index, row in preprocessed_data.iterrows():
#             driver_id = row['Driver']
#             lap_number = row['LapNumber']
#             actual = row['HasPitStop']
#             predicted = predicted_pitstops[index]

#             # Classify each lap
#             if actual == 1 and predicted == 1:
#                 categorized_pitstops[driver_id]["both"].append(lap_number)
#             elif actual == 1 and predicted == 0:
#                 categorized_pitstops[driver_id]["actual_only"].append(lap_number)
#             elif actual == 0 and predicted == 1:
#                 categorized_pitstops[driver_id]["predicted_only"].append(lap_number)

#         return {"pitstops": categorized_pitstops}
    
#     except Exception as e:
#         print(f"Error fetching pit stop comparison: {str(e)}")
#         return {"error": "Failed to load pit stop data"}, 500

@app.get("/events/{year}")
async def get_events(year: int):
    fastf1.Cache.enable_cache("cache\F1 cache")
    schedule = fastf1.get_event_schedule(year)
    event_list = [{"EventName": event.EventName, "RoundNumber": event.RoundNumber} for event in schedule.itertuples()]
    return {"events": event_list}

# Endpoint to fetch track information (X, Y coordinates)
@app.get("/track/{year}/{event_name}")
async def get_track_info(year: int, event_name: str):
    try:
        fastf1.Cache.enable_cache("cache\F1 cache")
        schedule = fastf1.get_event_schedule(year)
        event = schedule.loc[schedule['EventName'] == event_name].iloc[0]
        session = fastf1.get_session(year, event['RoundNumber'], 'R')
        session.load()
        if session.laps.empty:
            raise Exception("No laps available in session")
        fastest_lap = session.laps.pick_fastest()
        pos_data = fastest_lap.get_pos_data()
        track_data = [{"X": float(x), "Y": float(y)} for x, y in pos_data[['X', 'Y']].to_numpy()]
        return {"track": track_data}
    except Exception as e:
        print(f"Error fetching track info: {str(e)}")
        return {"error": "Failed to load track info"}, 500

# Endpoint to fetch driver position data
@app.get("/race/{year}/{event_name}/positions")
async def get_driver_positions(year: int, event_name: str):
    try:
        fastf1.Cache.enable_cache("cache\F1 cache")
        schedule = fastf1.get_event_schedule(year)
        event = schedule.loc[schedule['EventName'] == event_name].iloc[0]
        session = fastf1.get_session(year, event['RoundNumber'], 'R')
        session.load()
        position_data = session.pos_data
        driver_positions = {}
        for driver_num, pos_df in position_data.items():
            positions = pos_df[['X', 'Y', 'Time']].to_dict(orient='records')
            driver_positions[driver_num] = positions
        return {"positions": driver_positions}
    except Exception as e:
        print(f"Error fetching driver positions: {str(e)}")
        return {"error": "Failed to load driver positions"}, 500

# Endpoint to fetch driver abbreviations
@app.get("/race/{year}/{event_name}/drivers")
async def get_driver_abbreviations(year: int, event_name: str):
    try:
        fastf1.Cache.enable_cache("cache\F1 cache")
        schedule = fastf1.get_event_schedule(year)
        event = schedule.loc[schedule['EventName'] == event_name].iloc[0]
        session = fastf1.get_session(year, event['RoundNumber'], 'R')
        session.load()
        driver_abbreviations = {}
        for driver_num in session.drivers:
            driver_info = session.get_driver(driver_num)
            driver_abbreviations[driver_num] = driver_info.Abbreviation
        return {"drivers": driver_abbreviations}
    except Exception as e:
        print(f"Error fetching driver abbreviations: {str(e)}")
        return {"error": "Failed to load driver abbreviations"}, 500

@app.get("/race/{year}/{event_name}/stats")
async def get_driver_stats(year: int, event_name: str):
    try:
        fastf1.Cache.enable_cache("cache\F1 cache")
        schedule = fastf1.get_event_schedule(year)
        event = schedule.loc[schedule['EventName'] == event_name].iloc[0]
        session = fastf1.get_session(year, event['RoundNumber'], 'R')
        session.load()
        driver_stats = []
        for driver_num in session.drivers:
            driver = session.get_driver(driver_num)
            lap_times = session.laps.pick_driver(driver_num)['LapTime'].dropna().to_list()
            current_tyre = session.laps.pick_driver(driver_num)['Compound'].mode()[0] if not session.laps.pick_driver(driver_num)['Compound'].empty else "Unknown"
            driver_stats.append({
                "Position": session.results.loc[driver_num]['Position'],
                "Driver": driver.Abbreviation,
                "LapTime": str(lap_times[-1]) if lap_times else "N/A",
                "CurrentTyre": current_tyre
            })
        return {"stats": driver_stats}
    except Exception as e:
        print(f"Error fetching driver stats: {str(e)}")
        return {"error": "Failed to load driver stats"}, 500

# # WebSocket for real-time updates
# @app.websocket("/ws/race/{year}/{event_name}")
# async def websocket_driver_positions(websocket: WebSocket, year: int, event_name: str):
#     await websocket.accept()
#     schedule = fastf1.get_event_schedule(year)
#     event = schedule.loc[schedule['EventName'] == event_name].iloc[0]
#     session = fastf1.get_session(year, event['RoundNumber'], 'R')
#     session.load()
#     position_data = session.pos_data
#     drivers = position_data.keys()
#     while True:
#         driver_positions = {}
#         for driver_num in drivers:
#             positions = position_data[driver_num][['X', 'Y', 'Time']].to_dict(orient='records')
#             driver_positions[driver_num] = positions
#         await websocket.send_json(driver_positions)
