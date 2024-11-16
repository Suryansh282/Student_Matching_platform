from flask import Flask, request, render_template, jsonify
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import pairwise_distances
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models and encoders
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
mlb = joblib.load('mlb.pkl')

# Load student data
file_path = 'students_large_dataset.json'
with open(file_path, 'r') as file:
    students = json.load(file)

# Convert JSON data to DataFrame
student_df = pd.DataFrame(students)
student_df['subjects'] = student_df['subjects'].apply(lambda x: [s.lower() for s in x])

# Define goal mapping (mapping study goals to numeric values)
goal_mapping = {"Exam preparation": 0, "Assignments": 1, "Concept review": 2, "Project work": 3, "Homework": 4, "Practice": 5}

# Add cluster predictions to DataFrame
student_df['cluster'] = kmeans_model.predict(
    scaler.transform(
        np.hstack([
            mlb.transform(student_df['subjects']),
            student_df['study_hours'].apply(np.mean).values.reshape(-1, 1),
            student_df['goals'].map(goal_mapping).values.reshape(-1, 1)
        ])
    )
)

def find_matching_students(new_student, student_df, kmeans, scaler, mlb, n_matches=3):
    """
    Find students who match the study profile of a new student.

    Args:
        new_student (dict): The new student's information including subjects, study hours, and goals.
        student_df (pd.DataFrame): DataFrame containing student data with their features and cluster information.
        kmeans (sklearn.cluster.KMeans): Pre-trained KMeans clustering model to predict the cluster of the new student.
        scaler (sklearn.preprocessing.StandardScaler): Pre-trained StandardScaler for scaling features.
        mlb (sklearn.preprocessing.MultiLabelBinarizer): Pre-trained MultiLabelBinarizer for encoding subjects.
        n_matches (int): Number of closest matching students to return. Default is 3.

    Returns:
        pd.DataFrame: DataFrame containing the n closest matching students.
    """
    # Process new student data
    new_subjects = mlb.transform([[s.lower() for s in new_student["subjects"]]])[0]
    new_avg_hour = np.mean(new_student["study_hours"])
    new_goal_encoded = goal_mapping[new_student["goals"]]
    
    # Combine features for the new student
    new_student_features = np.hstack([new_subjects, new_avg_hour, new_goal_encoded]).reshape(1, -1)
    new_student_features_scaled = scaler.transform(new_student_features)
    
    # Predict the cluster for the new student
    cluster = kmeans.predict(new_student_features_scaled)[0]
    
    # Filter students from the same cluster
    cluster_students = student_df[student_df["cluster"] == cluster]
    
    # Prepare features for students in the same cluster
    cluster_features = scaler.transform(
        np.hstack([
            mlb.transform(cluster_students['subjects']),
            cluster_students['study_hours'].apply(np.mean).values.reshape(-1, 1),
            cluster_students['goals'].map(goal_mapping).values.reshape(-1, 1)
        ])
    )
    
    # Calculate distances between the new student and cluster students
    distances = pairwise_distances(new_student_features_scaled, cluster_features, metric="euclidean").flatten()
    
    # Find the closest matches
    closest_indices = distances.argsort()[:n_matches]
    
    return cluster_students.iloc[closest_indices]

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Route for handling the main page where users can input a new student's study data 
    and find matching study partners.

    If the request method is POST, it processes the submitted data, finds matching students,
    and renders the results page.

    Returns:
        str: Rendered HTML page (either index.html or results.html).
    """
    if request.method == 'POST':
        # Collect new student's data from the form
        new_student = {
            "name": request.form['name'],
            "subjects": request.form['subjects'].split(','),
            "study_hours": list(map(int, request.form['study_hours'].split(','))),
            "goals": request.form['goals']
        }
        
        # Find matching students based on the new student's data
        matches = find_matching_students(new_student, student_df, kmeans_model, scaler, mlb, n_matches=5)
        
        # Render the results page with the matches
        return render_template('results.html', matches=matches.to_dict(orient='records'))
    
    # Render the index page if request method is GET
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
