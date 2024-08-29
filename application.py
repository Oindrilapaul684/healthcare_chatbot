import streamlit as st
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import os
import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load datasets
training = pd.read_csv('Training (1).csv')
testing = pd.read_csv('Testing (1).csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)

model = SVC()
model.fit(x_train, y_train)

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()
symptoms_dict = {symptom: index for index, symptom in enumerate(x.columns)}

def getDescription():
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

def getSeverityDict():
    with open('Symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            severityDictionary[row[0]] = int(row[1])

def getprecautionDict():
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

def sec_predict(symptoms_exp):
    df = pd.read_csv('Training (1).csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1

    return rf_clf.predict([input_vector])

def calc_condition(exp, days):
    sum_severity = sum(severityDictionary[item] for item in exp)
    if (sum_severity * days) / (len(exp) + 1) > 13:
        st.write("You should take the consultation from a doctor.")
    else:
        st.write("It might not be that bad, but you should take precautions.")

def tree_to_code(tree, feature_names, disease_input, num_days):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            symptoms_given = cols[reduced_data.loc[present_disease].values[0].nonzero()] # type: ignore

            symptoms_exp = []
            for syms in list(symptoms_given):
                response = st.radio(f"Are you experiencing {syms}?", ("yes", "no"))
                if response == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                st.write(f"You may have {present_disease[0]}")
                st.write(description_list[present_disease[0]])
            else:
                st.write(f"You may have {present_disease[0]} or {second_prediction[0]}")
                st.write(description_list[present_disease[0]])
                st.write(description_list[second_prediction[0]])

            st.write("Take the following measures:")
            for i, j in enumerate(precautionDictionary[present_disease[0]]):
                st.write(f"{i+1}) {j}")

    recurse(0, 1)

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def remove_past_appointments():
    if os.path.exists('appointments.csv'):
        current_date = datetime.date.today()
        updated_appointments = []

        with open('appointments.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                appointment_date = datetime.datetime.strptime(row[1], "%Y-%m-%d").date()
                if appointment_date >= current_date:
                    updated_appointments.append(row)

        with open('appointments.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(updated_appointments)
    else:
        st.write("No appointments to remove.")

def schedule_appointment():
    with st.form(key="schedule_form"):
        st.write("Please enter the following details to schedule your appointment:")
        name = st.text_input("Your Name:")
        date_input = st.date_input("Preferred Date:")
        time_input = st.time_input("Preferred Time:")
        doctor_name = st.text_input("Doctor's Name:")
        location = st.text_input("Location:")
        submit_button = st.form_submit_button(label="Schedule Appointment")

        if submit_button:
            try:
                with open('appointments.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, date_input, time_input, doctor_name, location])
                st.success(f"Appointment scheduled successfully for {name} with Dr. {doctor_name} on {date_input} at {time_input} in {location}.")
            except ValueError:
                st.error("Invalid date or time format. Please try again.")

def show_appointment_list():
    remove_past_appointments()
    if os.path.exists('appointments.csv'):
        st.write("List of Scheduled Appointments:")
        with open('appointments.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                st.write(f"Name: {row[0]}, Date: {row[1]}, Time: {row[2]}, Doctor: {row[3]}, Location: {row[4]}")
    else:
        st.write("No appointments have been scheduled yet.")

def main():
    st.title("HealthCare ChatBot")

    getDescription()
    getSeverityDict()
    getprecautionDict()

    st.sidebar.title("Menu")
    user_action = st.sidebar.radio("Choose an action:", 
                                   ("Predict Disease", "Schedule Appointment", "View Appointments"))

    if user_action == "Predict Disease":
        st.header("Disease Prediction")
        disease_input = st.text_input("Enter the symptom you are experiencing:")
        num_days = st.number_input("How many days have you been experiencing this symptom?", min_value=1, step=1)
        if st.button("Predict"):
            if disease_input:
                tree_to_code(clf, cols, disease_input, num_days)
            else:
                st.write("Please enter a symptom.")
    elif user_action == "Schedule Appointment":
        st.header("Schedule Appointment")
        schedule_appointment()
    elif user_action == "View Appointments":
        st.header("Scheduled Appointments")
        show_appointment_list()

if __name__ == "__main__":
    main()
