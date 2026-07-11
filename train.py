import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Load and clean the dataset (looks directly in the current folder)
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.dropna(how="any", axis=1)
df.columns = ['label', 'message']
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

# 3. Vectorize and Train Model
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = tfidf.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("🎉 Model trained successfully!")

# 4. Save the model and vectorizer directly in the current folder
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

print("💾 'model.pkl' and 'tfidf.pkl' saved successfully in the current directory.")