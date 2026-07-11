import pickle
import streamlit as st

# Set up page configuration
st.set_page_config(page_title="Spam SMS Detector", page_icon="🛡️", layout="centered")

# 1. Load the pre-trained model and vectorizer from the current folder
@st.cache_resource
def load_assets():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf.pkl', 'rb') as tfidf_file:
        tfidf = pickle.load(tfidf_file)
    return model, tfidf

try:
    model, tfidf = load_assets()
except FileNotFoundError:
    st.error("❌ 'model.pkl' or 'tfidf.pkl' not found! Please run 'train.py' first to generate them.")
    st.stop()

# 2. Design the Streamlit UI
st.title("🛡️ Spam SMS Detector")
st.write("Type any message below to see if the trained Naïve Bayes model flags it as Spam or Safe (Ham).")

# Text area input
message = st.text_area("SMS Message", placeholder="Type or paste your SMS content here...", height=150)

# Predict button
if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        # Transform and predict
        text_tfidf = tfidf.transform([message])
        prediction = model.predict(text_tfidf)
        probability = model.predict_proba(text_tfidf)[0]
        
        # Format metrics
        result = "🚨 SPAM" if prediction[0] == 1 else "✅ Safe (Ham)"
        confidence = probability[prediction[0]] * 100
        
        # Display Results beautifully
        st.subheader("Model Assessment")
        if prediction[0] == 1:
            st.error(f"**Prediction:** {result}")
        else:
            st.success(f"**Prediction:** {result}")
            
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

# 3. Add clickable examples sidebar
st.sidebar.markdown("### 💡 Quick Examples")
if st.sidebar.button("📋 Copy Spam Example"):
    st.info("Copy this: *CONGRATULATIONS! You have won a £1,000 Walmart Gift Card. Click here to claim.*")
if st.sidebar.button("📋 Copy Safe Example"):
    st.info("Copy this: *Hey mom, I'm running a bit late for dinner. Save some food for me!*")
