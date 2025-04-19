import gradio as gr
import joblib

# Load the model pipeline
pipeline = joblib.load('cat_cla.pkl')

# Define the prediction function
def predict(input_text):
    return pipeline.predict([input_text])[0]  # wrap input in list & get scalar result

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter input text"),
    outputs=gr.Label(label="Predicted class")
)

if __name__ == "__main__":
    demo.launch()
