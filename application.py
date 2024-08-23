import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
import datetime
import os
import csv

# Load datasets
training = pd.read_csv('Training (1).csv')
testing = pd.read_csv('Testing (1).csv')

# Prepare data
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = le.transform(testing['prognosis'])

# Train model
clf = DecisionTreeClassifier().fit(x_train, y_train)

# Load dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {symptom: index for index, symptom in enumerate(x)}

def load_dictionaries():
    global severityDictionary, description_list, precautionDictionary
    
    with open('Symptom_severity.csv') as file:
        reader = csv.reader(file)
        severityDictionary = {rows[0]: int(rows[1]) for rows in reader}
    
    with open('symptom_Description.csv') as file:
        reader = csv.reader(file)
        description_list = {rows[0]: rows[1] for rows in reader}
    
    with open('symptom_precaution.csv') as file:
        reader = csv.reader(file)
        precautionDictionary = {rows[0]: rows[1:] for rows in reader}

def predict_disease(symptoms_exp):
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1
    return clf.predict([input_vector])

# Streamlit app
st.title("Healthcare ChatBot")

# Get user's name
name = st.text_input("Enter your name")
st.write(f"Hello, {name}!")

# Symptoms input
symptom_input = st.text_input("Enter your symptom")

# Days experiencing symptom
days = st.number_input("How many days have you experienced this?", min_value=0, max_value=100, value=1)

if st.button("Predict"):
    if symptom_input:
        if symptom_input in symptoms_dict:
            symptoms_exp = [symptom_input]
            result = predict_disease(symptoms_exp)
            disease = le.inverse_transform(result)[0]
            
            st.write(f"You may have: **{disease}**")
            st.write(description_list[disease])
            st.write("Take the following precautions:")
            for precaution in precautionDictionary[disease]:
                st.write(f"- {precaution}")
            
            if (severityDictionary[symptom_input] * days) / (len(symptoms_exp) + 1) > 13:
                st.write("You should consult a doctor.")
            else:
                st.write("It's recommended to take precautions.")
        else:
            st.write("Please enter a valid symptom.")
    else:
        st.write("Symptom input cannot be empty.")

# Appointment scheduling
st.header("Schedule an Appointment")
if st.button("Schedule Appointment"):
    appointment_date = st.date_input("Preferred Date")
    appointment_time = st.time_input("Preferred Time")
    
    if name and appointment_date and appointment_time:
        with open('appointments.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, appointment_date, appointment_time])
        st.write(f"Appointment scheduled for {name} on {appointment_date} at {appointment_time}.")

# Show appointments
if st.button("Show Appointments"):
    if os.path.exists('appointments.csv'):
        st.write("Scheduled Appointments:")
        with open('appointments.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                st.write(f"Name: {row[0]}, Date: {row[1]}, Time: {row[2]}")
    else:
        st.write("No appointments scheduled yet.")

# Load dictionaries when the app starts
load_dictionaries()
