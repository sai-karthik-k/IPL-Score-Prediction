# IPL Score Prediction Model

This repository contains a machine learning model built using Keras for predicting the scores in Indian Premier League (IPL) matches. The model is trained using historical IPL match data and is deployed using a Streamlit application for user interaction.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Streamlit App](#streamlit-app)
- [Screenshot](#Screenshot)

## Overview

The goal of this project is to predict the total score a team might achieve in an IPL match based on input features such as team, venue, innings, and other factors. The model is trained using Keras, and the predictions are made accessible through a web interface built using Streamlit.

## Features
- **Data Preprocessing**: IPL match data is preprocessed for model training.
- **Model Training**: The Keras model is trained on the processed dataset.
- **Score Prediction**: Users can input match details to predict the final score.
- **Streamlit Integration**: The app provides a user-friendly interface for input and displays the predictions.
  
## Technologies
- Python 3.x
- Keras for building and training the machine learning model
- Pandas, NumPy for data processing
- Matplotlib, Seaborn for data visualization
- Streamlit for app deployment

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/sai-karthik-k/IPL-Score-Prediction.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. To run the Streamlit app, use the command:
    ```bash
    streamlit run app.py
    ```

## Usage

After running the Streamlit app, a web interface will open where you can enter match details such as teams, venue, and other relevant factors. The app will use the trained model to predict the total score of the match.

## Model Training

The model is built using a Keras sequential model and is trained on historical IPL match data. The steps involved include:

1. **Data Loading**: Loading IPL match data for preprocessing.
2. **Data Preprocessing**: Handling missing values, encoding categorical features, and normalizing the data.
3. **Model Training**: Training a deep learning model using Keras.
4. **Evaluation**: Testing the model using validation data to evaluate its performance.

The detailed training process is included in the `IPL_Score_Prediction.py` file.

## Streamlit App

The app is built using Streamlit, which provides an intuitive interface for users to interact with the model and predict match scores. The `app.py` file contains the Streamlit application code.

## Screenshot
![Uploading image.png…]()

