## IPL Score Prediction using Machine Learning

This repository contains a Python project for predicting IPL scores using a neural network model. 

**Project Structure**
**Data**

The project uses the `ipl_data.csv` dataset, which contains information about IPL matches, including:

- `mid`: Match ID
- `date`: Date of the match
- `venue`: Venue of the match
- `bat_team`: Batting team
- `bowl_team`: Bowling team
- `batsman`: Batsman
- `bowler`: Bowler
- `runs`: Runs scored in the match
- `wickets`: Wickets taken in the match
- `overs`: Overs bowled in the match
- `runs_last_5`: Runs scored in the last 5 overs
- `wickets_last_5`: Wickets taken in the last 5 overs
- `striker`: Striker
- `non-striker`: Non-striker
- `total`: Total score of the match

**Model**

The project implements a neural network model using the Keras library. The model is trained using a Huber loss function to predict the total score of a match based on the given features.

**Features**

- Data preprocessing and feature engineering
- Label encoding for categorical features
- Feature scaling using MinMaxScaler
- Training a neural network model using Keras
- Evaluation of the model using Mean Absolute Error (MAE)
- Interactive widget for predicting scores using user inputs

**Usage**

1. Clone the repository.
2. Install the necessary libraries: `pip install -r requirements.txt`
3. Run the `ipl_score_prediction.ipynb` notebook.

**Contributing**

Contributions are welcome! Please feel free to open an issue or submit a pull request.



**Run Video**

![Project Run](project_run.gif)
