import streamlit as st
import numpy as np
import pickle
import xgboost as xgb

# Load the trained XGBoost model
model_path = "models/xgboost_best_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Define the feature columns
feature_columns = ['age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 
                   'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 
                   'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']

# Streamlit UI
st.set_page_config(page_title="Parkinson's Disease Prediction", layout="wide")
st.markdown("<h1 style='text-align: center; background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet); -webkit-background-clip: text; color: transparent;'>Parkinson's Disease Motor UPDRS Prediction</h1>", unsafe_allow_html=True)
st.write("Use the sliders below to input feature values and predict motor UPDRS.")

# Create columns for better layout
col1, col2 = st.columns(2)

# Collect user inputs using sliders
input_values = {}
def get_slider_value(feature, min_val, max_val, step, default, col):
    return col.slider(f"{feature}", min_val, max_val, default, step)

input_values['age'] = get_slider_value('Age', 30, 90, 1, 60, col1)
input_values['sex'] = col1.radio("Sex", [0, 1], index=0, format_func=lambda x: "Male" if x == 1 else "Female")
input_values['test_time'] = get_slider_value('Test Time', 0, 200, 1, 100, col1)

for i, feature in enumerate(feature_columns[3:]):  # Exclude age, sex, and test_time handled above
    col = col1 if i % 2 == 0 else col2
    input_values[feature] = get_slider_value(feature, 0.0, 1.0, 0.01, 0.5, col)

# Convert input values into a NumPy array for prediction
input_array = np.array([list(input_values.values())]).reshape(1, -1)

# Predict button
if st.button("Predict UPDRS"):
    prediction = model.predict(input_array)[0]
    st.success(f"Predicted Motor UPDRS: {prediction:.2f}")
