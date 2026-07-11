# 🛡️ Spam SMS Detector

An interactive machine learning web application that uses a Naïve Bayes classifier to detect spam SMS messages.

## 🛠️ Built With

* **Pandas** - For data loading and handling
* **Scikit-Learn** - For TF-IDF text vectorization and the Naïve Bayes Machine Learning model
* **Gradio** - For creating the interactive web UI

## 📂 Flat Project Structure

All files must live directly in the same root folder (no subdirectories required):

```text
├── spam.csv            # Your original SMS dataset
├── train.py            # Model training and serialization script
├── app.py              # Gradio UI application web script
├── requirements.txt    # Python package dependencies
├── model.pkl           # Generated automatically after running train.py
└── tfidf.pkl           # Generated automatically after running train.py