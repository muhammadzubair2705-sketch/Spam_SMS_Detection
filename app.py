import pickle
import gradio as gr

# 1. Load the pre-trained model and vectorizer from the current folder
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

print("🔌 Model and Vectorizer loaded successfully! Launching UI...")

# 2. Define the prediction function
def predict_sms(message):
    if not message.strip():
        return "Please enter some text."
    
    # Transform the input text using the loaded TF-IDF vectorizer
    text_tfidf = tfidf.transform([message])
    
    # Predict using the loaded Naïve Bayes model
    prediction = model.predict(text_tfidf)
    probability = model.predict_proba(text_tfidf)[0] # Get confidence scores
    
    # Format the output based on prediction class (0 = Safe, 1 = Spam)
    result = "🚨 SPAM" if prediction[0] == 1 else "Safe"
    confidence = probability[prediction[0]] * 100
    
    return f"Prediction: {result}\nConfidence: {confidence:.2f}%"

# 3. Create the Gradio Interface
interface = gr.Interface(
    fn=predict_sms, 
    inputs=gr.Textbox(lines=3, placeholder="Type or paste your SMS content here...", label="SMS Message"),
    outputs=gr.Textbox(label="Model Assessment"),
    title="🛡️ Spam SMS Detector",
    description="Type any message below to see if the trained Naïve Bayes model flags it as Spam or Safe (Ham).",
    examples=[
        ["CONGRATULATIONS! You have won a £1,000 Walmart Gift Card. Click here to claim."],
        ["Hey mom, I'm running a bit late for dinner. Save some food for me!"]
    ]
)

# Launch the app with sharing and debugging enabled for cloud deployment
if __name__ == "__main__":
    interface.launch(share=True, debug=True)
