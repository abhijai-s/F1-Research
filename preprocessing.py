import pandas as pd
import joblib
import numpy as np
# Load necessary preprocessing tools, e.g., scaler and imputer
scaler1 = joblib.load("models/scaler1v2.pkl")
scaler2 = joblib.load("models/scaler2v2.pkl")
scaler3 = joblib.load("models/scaler_modelv2.pkl")
imputer = joblib.load("models/imputer1v2.pkl")

def preprocess_lap_data(df, event_name):
    print("Initial dataframe in preprocessing:", df.head())
    df = df[['Driver', 'DriverNumber', 'LapTime', 'LapNumber', 'Stint', 'PitOutTime', 'PitInTime',
             'Compound', 'TyreLife', 'Team', 'TrackStatus', 'Position']].copy()
    # Step 1: Drop rows with certain tire compounds
    df = df[~df['Compound'].isin(['INTERMEDIATE', 'WET', 'UNKNOWN'])].reset_index(drop=True)
    print(df.Compound.value_counts())
     # Step 2: Use existing 'LapTime' as 'LapTime_Seconds' directly
    df['LapTime_Seconds'] = pd.to_timedelta(df['LapTime'], errors='coerce').dt.total_seconds()
    # df['LapTime_Seconds'] = pd.to_numeric(df['LapTime_Seconds'], errors='coerce')  # Ensure numeric type
    print("Using LapTime directly as LapTime_Seconds:", df[['LapTime', 'LapTime_Seconds']].head())

    # Ensure 'LapTime_Seconds' is float
    print("Data types before scaling:", df[['Position', 'LapTime_Seconds']].dtypes)

    # Step 3: Apply scaling and KNN imputation
    df[['Position','LapTime_Seconds']] = scaler1.transform(df[['Position','LapTime_Seconds']])
    df[['Position','LapTime_Seconds']] = imputer.transform(df[['Position','LapTime_Seconds']])
    df[['Position','LapTime_Seconds']] = scaler1.inverse_transform(df[['Position','LapTime_Seconds']])
    print("After scaling and imputing Position:", df[['Position','LapTime_Seconds']].head())

    # Step 4: Convert 'PitInTime' and 'PitOutTime' to seconds
    # df['PitInTime_Seconds'] = df['PitInTime'].apply(lambda x: 0 if pd.isnull(x) else pd.to_timedelta(x).total_seconds())
    # df['PitOutTime_Seconds'] = df['PitOutTime'].apply(lambda x: 0 if pd.isnull(x) else pd.to_timedelta(x).total_seconds())

    df['PitInTime_Seconds'] = pd.to_timedelta(df['PitInTime'].fillna(pd.Timedelta(seconds=0))).dt.total_seconds()
    df['PitOutTime_Seconds'] = pd.to_timedelta(df['PitOutTime'].fillna(pd.Timedelta(seconds=0))).dt.total_seconds()
    print("After converting PitInTime and PitOutTime to seconds:", df[['PitInTime', 'PitInTime_Seconds', 'PitOutTime', 'PitOutTime_Seconds']].head())

    # Step 5: Fill missing 'TrackStatus' values with 1
    df['TrackStatus'].fillna(1, inplace=True)

    # Step 6: Create CumulativeTimeStint
    df['CumulativeTimeStint'] = df.groupby([ 'Driver', 'Stint'])['LapTime_Seconds'].cumsum()

    # Step 7: Sort and create 'DriverAheadPit' and 'DriverBehindPit' features
    df = df.sort_values(by=['LapNumber', 'Stint', 'Position'])
    df['DriverAheadPit'] = df.groupby(['LapNumber', 'Stint'])['PitInTime'].shift(-1).notnull().astype(int)
    df['DriverBehindPit'] = df.groupby([ 'LapNumber', 'Stint'])['PitInTime'].shift(1).notnull().astype(int)
    df = df.sort_values(by=[ 'Driver', 'LapNumber']).reset_index(drop=True)

    ##modification(extra feature engineering for new data)
    #1.
    df['delta_laptime'] = df.groupby('Driver')['LapTime_Seconds'].diff()
    df['delta_laptime'] = df['delta_laptime'].fillna(0)
    #2.
    df['max_lap_for_event'] = df['LapNumber'].max()
    df['race_progress_fraction'] = df['LapNumber'] / df['max_lap_for_event']
    df['race_progress_fraction'] = df['race_progress_fraction'].fillna(0)

    ##mod ends

    df[['LapNumber', 'Stint', 'TyreLife']] = df[['LapNumber', 'Stint', 'TyreLife']].astype('int64')
    df['DriverNumber'] = pd.to_numeric(df['DriverNumber'], errors='coerce').astype('int64')
    # Step 8: One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['Compound', 'Driver', 'Team'], dtype=int, drop_first=True)

    # Step 9: Encode 'TrackStatus' (1 to 7) as binary columns
    df['TrackStatus'] = df['TrackStatus'].astype(str)
    for i in range(1, 8):
        df[f'TrackStatus_{i}'] = df['TrackStatus'].apply(lambda x: 1 if str(i) in x else 0)

    # Step 10: Add 'HasPitStop' target feature
    df['HasPitStop'] = df['PitInTime_Seconds'].apply(lambda x: 0 if x == 0 else 1)
    print(df.HasPitStop.value_counts())

     # Step 11: Dynamically create the event-specific column
    event_column_name = f'EventName_{event_name}'
    df[event_column_name] = 1  # Set the event column to 1 for all rows

    # Step 11 & 12: Ensure all necessary columns are present
    required_columns = ['EventName_Abu Dhabi Grand Prix',
       'EventName_Australian Grand Prix', 'EventName_Austrian Grand Prix',
       'EventName_Azerbaijan Grand Prix', 'EventName_Bahrain Grand Prix',
       'EventName_Belgian Grand Prix', 'EventName_British Grand Prix',
       'EventName_Canadian Grand Prix', 'EventName_Chinese Grand Prix',
       'EventName_Dutch Grand Prix', 'EventName_Eifel Grand Prix',
       'EventName_Emilia Romagna Grand Prix', 'EventName_French Grand Prix',
       'EventName_Hungarian Grand Prix', 'EventName_Italian Grand Prix',
       'EventName_Japanese Grand Prix', 'EventName_Las Vegas Grand Prix',
       'EventName_Mexico City Grand Prix', 'EventName_Miami Grand Prix',
       'EventName_Monaco Grand Prix', 'EventName_Portuguese Grand Prix',
       'EventName_Qatar Grand Prix', 'EventName_Russian Grand Prix',
       'EventName_Sakhir Grand Prix', 'EventName_Saudi Arabian Grand Prix',
       'EventName_Singapore Grand Prix', 'EventName_Spanish Grand Prix',
       'EventName_Styrian Grand Prix', 'EventName_São Paulo Grand Prix',
       'EventName_Turkish Grand Prix', 'EventName_Tuscan Grand Prix',
       'EventName_United States Grand Prix','Team_Alfa Romeo Racing', 'Team_AlphaTauri', 'Team_Alpine',
       'Team_Aston Martin', 'Team_Ferrari', 'Team_Haas F1 Team',
       'Team_Kick Sauber', 'Team_McLaren', 'Team_Mercedes', 'Team_RB',
       'Team_Racing Point', 'Team_Red Bull Racing', 'Team_Renault',
       'Team_Williams','DriverNumber', 'Driver_ALB', 'Driver_ALO',
       'Driver_BEA', 'Driver_BOT', 'Driver_COL', 'Driver_DEV', 'Driver_FIT',
       'Driver_GAS', 'Driver_GIO', 'Driver_GRO', 'Driver_HAM', 'Driver_HUL',
       'Driver_KUB', 'Driver_KVY', 'Driver_LAT', 'Driver_LAW', 'Driver_LEC',
       'Driver_MAG', 'Driver_MAZ', 'Driver_MSC', 'Driver_NOR', 'Driver_OCO',
       'Driver_PER', 'Driver_PIA', 'Driver_RAI', 'Driver_RIC', 'Driver_RUS',
       'Driver_SAI', 'Driver_SAR', 'Driver_STR', 'Driver_TSU', 'Driver_VER',
       'Driver_VET', 'Driver_ZHO','LapNumber', 'LapTime_Seconds','Position','Stint','TrackStatus_1', 'TrackStatus_2', 'TrackStatus_3',
       'TrackStatus_4', 'TrackStatus_5', 'TrackStatus_6', 'TrackStatus_7','Compound_MEDIUM', 'Compound_SOFT', 'CumulativeTimeStint',
       'DriverAheadPit','DriverBehindPit','HasPitStop','TyreLife','delta_laptime','race_progress_fraction']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    # Step 13: Scale features
    features_to_normalize = [
        'LapTime_Seconds', 'CumulativeTimeStint','delta_laptime','race_progress_fraction',
        'TyreLife', 'LapNumber', 'Position', 'Stint','DriverNumber'
    ]
    df[features_to_normalize] = scaler2.transform(df[features_to_normalize])

    print("After scaling final features:", df[features_to_normalize].head())
    df2 = df.to_csv('preprocessed_data.csv', index=False)
    # Select the final set of features
    df = df[required_columns]
    print("Final preprocessed data for model input:", df.head())
    X = df.drop(columns=['HasPitStop'])
    y = df['HasPitStop']
    X_scaled = scaler3.transform(X)

    # Set sequence length for data (e.g., timesteps = 10)
    timesteps = 10

    # Create sequences for X_scaled
    def create_sequences(X, y, timesteps,step =1):
        Xs, ys = [], []
        for i in range(len(X) - timesteps):
            Xs.append(X[i:i+timesteps])
            ys.append(y[i+timesteps-1])
        return np.array(Xs), np.array(ys)

    # Apply to your data
    X_seq, y_seq = create_sequences(X_scaled, y, timesteps=timesteps, step =1)

    return X_seq, y_seq  # Return reshaped input and target for LSTM
    # return X_scaled , y
   
