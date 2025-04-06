# Ads-click-through-rate-prediction

🌐 Deployment on Hugging Face Spaces : https://huggingface.co/spaces/smrup/Click-Through-Rate-Prediction-for-Advertisements

This app is designed to run seamlessly on Hugging Face Spaces. Just upload the following files:
app.py
ads_info.csv
requirements.txt
README.md

🚀 Ad Click Prediction App
An AI-Powered Predictor for Ad Click Behavior
This interactive web application uses machine learning to predict whether a user is likely to click on an advertisement based on their online behavior and demographics.

Built with:
🧠 Random Forest Classifier
📊 Plotly for insightful visualizations
🎨 Gradio UI for elegant and interactive deployment

🛠 Features
✅ Predicts if a user will click on an ad
✅ Accepts inputs like daily time on site, age, area income, internet usage, and gender
✅ Displays an interactive CTR box plot to visualize behavior trends
✅ Simple, responsive design optimized for recruiters and demo presentations

📂 Dataset
The model is trained on ads_info.csv, which contains:
Daily Time Spent on Site
Age
Area Income
Daily Internet Usage
Gender
Clicked on Ad (target label)


📘 Project Explanation: Ad Click Prediction App
This project is an AI-powered web application that predicts whether a user is likely to click on an online advertisement based on their behavioral and demographic data.

🎯 Objective
To demonstrate how machine learning can be used in digital marketing to:
Understand user interaction patterns
Predict click-through behavior
Optimize ad targeting strategies

🧠 How It Works
Dataset
The model is trained on a dataset (ads_info.csv) containing:
Daily time spent on the website
Age
Area income
Daily internet usage
Gender
Whether the ad was clicked (Clicked on Ad: Yes/No)

Preprocessing
Gender is converted into numeric format (Male: 1, Female: 0)
Features are selected and used to train a Random Forest Classifier

Prediction
The user enters their info (age, time on site, income, etc.)
The model predicts if that user is likely to click on an ad

The result is shown as either:
🟢 “User is likely to click on the Ad”
🔴 “User is not likely to click on the Ad”

Visualization
An interactive box plot shows how time spent on site relates to ad-clicking behavior, revealing key insights about user attention and interest.

🧪 Technologies Used
Python for data handling and ML
Scikit-learn for modeling
Gradio for the user interface
Plotly for interactive visualizations
