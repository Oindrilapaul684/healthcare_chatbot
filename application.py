import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load data (replace with your data loading logic)
@st.cache  # Cache data for performance optimization
def load_data():
    training_data = pd.read_csv("Training (1).csv")
    testing_data = pd.read_csv("Testing (1).csv")
    return training_data, testing_data

# Preprocess data (adapt based on your preprocessing steps)
def preprocess_data(data):
    cols = data.columns[:-1]
    x = data[cols]
    y = data["prognosis"]

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test, le, cols

# Build the decision tree model (adjust hyperparameters if needed)
def build_model(x_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf

# Define functions for symptom input, prediction, and explanation
def get_user_symptoms():
    symptoms = st.multiselect("Select symptoms you are experiencing:", cols) # type: ignore
    return symptoms

def predict_disease(clf, symptoms, le, cols):
    symptom_vector = np.zeros(len(cols))
    for symptom in symptoms:
        symptom_vector[[cols.index(symptom)]] = 1
    predicted_disease = le.inverse_transform(clf.predict([symptom_vector]))[0]
    return predicted_disease

def explain_disease(predicted_disease):
    # Replace with your logic to retrieve description from an external source (e.g., dictionary)
    description = "Disease description for " + predicted_disease + " is not yet available."
    return description

# Streamlit app structure
def main():
    st.title("Healthcare ChatBot")

    training_data, testing_data = load_data()
    x_train, x_test, y_train, y_test, le, cols = preprocess_data(training_data)
    clf = build_model(x_train, y_train)

    if st.button("Start Consultation"):
        user_symptoms = get_user_symptoms()
        if user_symptoms:
            predicted_disease = predict_disease(clf, user_symptoms, le, cols)
            st.write(f"**Predicted Disease:** {predicted_disease}")
            st.write(explain_disease(predicted_disease))
        else:
            st.write("Please select at least one symptom.")

    st.write("------------------------------------------------------------------")
    st.write("**Disclaimer:** This app is for informational purposes only and should not be used as a substitute for professional medical advice.")

if __name__ == "__main__":
    main()