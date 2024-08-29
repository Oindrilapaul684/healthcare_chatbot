import streamlit as st
import pandas as pd
import re
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import os
import datetime

# Load datasets
training = pd.read_csv('Training (1).csv')
testing = pd.read_csv('Testing (1).csv')

cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf = DecisionTreeClassifier().fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)

model = SVC()
model.fit(x_train, y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Load description, severity, and precaution data
def load_data():
    description_list = {}
    severityDictionary = {}
    precautionDictionary = {}

    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

    with open('Symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            severityDictionary[row[0]] = int(row[1])

    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

    return description_list, severityDictionary, precautionDictionary

description_list, severityDictionary, precautionDictionary = load_data()

def calc_condition(exp, days):
    sum_severity = sum(severityDictionary.get(item, 0) for item in exp)
    if (sum_severity * days) / (len(exp) + 1) > 13:
        return "You should take the consultation from a doctor."
    else:
        return "It might not be that bad but you should take precautions."

def sec_predict(symptoms_exp):
    df = pd.read_csv('Training (1).csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = feature_names
    symptoms_present = []

    disease_input = st.text_input("Enter the symptom you are experiencing:", "")

    if disease_input:
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            st.write("Searches related to input:")
            for num, it in enumerate(cnf_dis):
                st.write(f"{num}) {it}")
            if cnf_dis:
                conf_inp = st.number_input(f"Select the one you meant (0 - {len(cnf_dis) - 1}):", min_value=0, max_value=len(cnf_dis) - 1)
                disease_input = cnf_dis[conf_inp]

        num_days = st.number_input("Okay. From how many days?", min_value=1)

        def recurse(node, depth):
            indent = "  " * depth
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
                reduced_data = training.groupby(training['prognosis']).max()
                red_cols = reduced_data.columns
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

                st.write("Are you experiencing any of the following symptoms:")
                symptoms_exp = []
                for syms in list(symptoms_given):
                    inp = st.radio(f"{syms}?", ["yes", "no"])
                    if inp == "yes":
                        symptoms_exp.append(syms)

                second_prediction = sec_predict(symptoms_exp)
                condition = calc_condition(symptoms_exp, num_days)
                st.write(condition)

                if present_disease[0] == second_prediction[0]:
                    st.write(f"You may have {present_disease[0]}")
                    st.write(description_list.get(present_disease[0], "No description available"))
                else:
                    st.write(f"You may have {present_disease[0]} or {second_prediction[0]}")
                    st.write(description_list.get(present_disease[0], "No description available"))
                    st.write(description_list.get(second_prediction[0], "No description available"))

                precaution_list = precautionDictionary.get(present_disease[0], [])
                st.write("Take the following measures:")
                for i, j in enumerate(precaution_list):
                    st.write(f"{i + 1}) {j}")

        recurse(0, 1)

def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    return (1, pred_list) if pred_list else (0, [])

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

def schedule_appointment():
    st.write("Would you like to schedule an appointment?")

    if st.button("Yes"):
        name = st.text_input("Your Name:")
        date_input = st.text_input("Preferred Date (YYYY-MM-DD):")
        time_input = st.text_input("Preferred Time (HH:MM):")
        doctor_name = st.text_input("Doctor's Name:")
        location = st.text_input("Location:")

        if st.button("Submit"):
            try:
                appointment_date = datetime.datetime.strptime(date_input, "%Y-%m-%d").date()
                appointment_time = datetime.datetime.strptime(time_input, "%H:%M").time()

                with open('appointments.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, appointment_date, appointment_time, doctor_name, location])

                st.write(f"Appointment scheduled successfully for {name} with Dr. {doctor_name} on {appointment_date} at {appointment_time} in {location}.")
            except ValueError:
                st.write("Invalid date or time format. Please try again.")

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

def chatbot_interaction():
    st.title("HealthCare ChatBot")

    if st.button("Start Conversation"):
        st.write("Hi! I am your HealthCare Chatbot.")

    options = st.selectbox("What would you like to do today?", ["Select", "Predict disease based on symptoms", "Schedule an appointment", "View scheduled appointments"])

    if options == "Predict disease based on symptoms":
        st.write("Let's proceed with disease prediction.")
        tree_to_code(clf, cols)

    elif options == "Schedule an appointment":
        schedule_appointment()

    elif options == "View scheduled appointments":
        show_appointment_list()

if __name__ == "__main__":
    chatbot_interaction()
