import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.io as pio

# Load and preprocess data
df = pd.read_csv("ads_info.csv")
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Set Plotly theme
pio.templates.default = "plotly_white"

# Train model
X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Gender"]]
y = df["Clicked on Ad"]
model = RandomForestClassifier()
model.fit(X, y)

# Click-through rate visualization
def create_ctr_plot():
    fig = px.box(df, 
                 x="Daily Time Spent on Site",  
                 color="Clicked on Ad", 
                 title="Click Through Rate based on Time Spent on Site", 
                 color_discrete_map={'Yes': 'blue', 'No': 'red'})
    fig.update_traces(quartilemethod="exclusive")
    return fig.to_html(full_html=False)

# Prediction function
def predict_click(daily_time, age, income, internet_usage, gender):
    gender = 1 if gender == "Male" else 0
    features = np.array([[daily_time, age, income, internet_usage, gender]])
    prediction = model.predict(features)[0]
    return "🟢 User is likely to click on the Ad" if prediction == "Yes" else "🔴 User is not likely to click on the Ad"

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 Ad Click Prediction
    **An AI-Powered Predictor for Ad Click Behavior**

    This interactive app analyzes user behavior and predicts whether they are likely to click on an ad. Built with 💡 machine learning and an elegant UI for recruiters and data enthusiasts.
    """)

    with gr.Row():
        with gr.Column():
            daily_time = gr.Slider(0, 100, label="🕒 Daily Time Spent on Site (mins)")
            age = gr.Slider(10, 100, label="🎂 Age")
            income = gr.Slider(0, 200000, label="💰 Area Income ($)")
            internet_usage = gr.Slider(0, 500, label="🌐 Daily Internet Usage (mins)")
            gender = gr.Radio(["Male", "Female"], label="👤 Gender")
            predict_btn = gr.Button("🎯 Predict")
        with gr.Column():
            output = gr.Textbox(label="Prediction Result", lines=2)

    predict_btn.click(predict_click, inputs=[daily_time, age, income, internet_usage, gender], outputs=[output])

    with gr.Accordion("📊 Visual Insight: CTR by Time on Site", open=False):
        gr.HTML(create_ctr_plot())

# Launch the app
demo.launch()
