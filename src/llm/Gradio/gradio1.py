import gradio as gr
from transformers import pipeline

clf_ko = pipeline(
    "text-classification",
    model="tabularisai/multilingual-sentiment-analysis",
    device_map="auto",
    dtype="auto",
)


def predict_sentiment(text):
    result = clf_ko(text)

    label = result[0]["label"]
    score = result[0]["score"]

    return f"감정 : {label}\n확률 : {score:.3f}"


demo = gr.Interface(
    fn=predict_sentiment, inputs=["text"], outputs=["text"], api_name="predict"
)

demo.launch()
