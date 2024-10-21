# import streamlit as st
# import pandas as pd
# import numpy as np
# import time
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# # Load the saved model
# model = tf.keras.models.load_model('ipl_score_prediction_model.keras')

# # Load the preprocessed data
# ipl = pd.read_csv('ipl_data.csv')

# # Feature Engineering and Preprocessing
# df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)

# # Separate features and target
# X = df.drop(['total'], axis=1)
# y = df['total']

# # Label Encoding
# venue_encoder = LabelEncoder()
# batting_team_encoder = LabelEncoder()
# bowling_team_encoder = LabelEncoder()

# X['venue'] = venue_encoder.fit_transform(X['venue'])
# X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
# X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])

# # Feature Scaling
# # Separate numerical features for scaling
# numerical_features = ['venue', 'bat_team', 'bowl_team']
# X_num = X[numerical_features]
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X_num)

# # Streamlit App
# st.title('IPL Score Prediction App')

# # Input widgets
# venue = st.selectbox('Select Venue', ipl['venue'].unique())
# batting_team = st.selectbox('Select Batting Team', ipl['bat_team'].unique())
# bowling_team = st.selectbox('Select Bowling Team', ipl['bowl_team'].unique())

# # Prediction Button
# if st.button('Predict Score'):
#     # Transform input data
#     input_data = np.array([
#         venue_encoder.transform([venue])[0],
#         batting_team_encoder.transform([batting_team])[0],
#         bowling_team_encoder.transform([bowling_team])[0]
#     ])
#     input_data = input_data.reshape(1, 3)
#     input_data = scaler.transform(input_data)

#     # Start timer
#     start_time = time.time()

#     # Predict score
#     predicted_score = model.predict(input_data)[0][0]

#     # Calculate model run time
#     end_time = time.time()
#     run_time = end_time - start_time

#     # Display prediction and model run time
#     st.success(f'Predicted Score: {int(predicted_score)}')
#     st.info(f'Model Run Time: {run_time:.4f} seconds')

import streamlit as st
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the saved model
model = tf.keras.models.load_model('ipl_score_prediction_model.keras')

# Load the preprocessed data
ipl = pd.read_csv('ipl_data.csv')

# Feature Engineering and Preprocessing
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)

# Separate features and target
X = df.drop(['total'], axis=1)
y = df['total']

# Label Encoding
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()

X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])

# Feature Scaling
# Separate numerical features for scaling
numerical_features = ['venue', 'bat_team', 'bowl_team']
X_num = X[numerical_features]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_num)

# Streamlit App
st.title('IPL Score Prediction App')

# Set the background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://wallpaperaccess.com/full/3818759.jpg");
background-size: cover;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}

h1, p {
    color: white;
}

.stButton > button {
    background-color: red;
    color: white;
}

.st-cz {
    background-color: rgba(33, 195, 84, 0.3);
}

.st-d6 {
    background-color: rgba(28, 131, 225, 0.3);
}

.stVerticalBlock{
    background-color: rgba(0, 0, 0, 0.1);
}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Input widgets
venue = st.selectbox('Select Venue', ipl['venue'].unique())
batting_team = st.selectbox('Select Batting Team', ipl['bat_team'].unique())
bowling_team = st.selectbox('Select Bowling Team', ipl['bowl_team'].unique())

# Prediction Button
if st.button('Predict Score'):
    # Transform input data
    input_data = np.array([
        venue_encoder.transform([venue])[0],
        batting_team_encoder.transform([batting_team])[0],
        bowling_team_encoder.transform([bowling_team])[0]
    ])
    input_data = input_data.reshape(1, 3)
    input_data = scaler.transform(input_data)

    # Start timer
    start_time = time.time()

    # Predict score
    predicted_score = model.predict(input_data)[0][0]

    # Calculate model run time
    end_time = time.time()
    run_time = end_time - start_time

    # Display prediction and model run time
    st.success(f'Predicted Score: {int(predicted_score)}')
    st.info(f'Model Run Time: {run_time:.4f} seconds')