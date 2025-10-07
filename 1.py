# train_model.py
import pandas as pd
import numpy as np
import re, pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Data
true_df = pd.read_csv("C:\\Users\\Divyasree\\Downloads\\archive (5)\\News _dataset\\True.csv")
fake_df = pd.read_csv("C:\\Users\\Divyasree\\Downloads\\archive (5)\\News _dataset\\Fake.csv")

# Label Encoding
true_df["label"] = 1   # 1 = REAL
fake_df["label"] = 0

# Combine & Shuffle
df = pd.concat([true_df, fake_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Clean Text Function
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text)) # remove URLs
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text) # remove non-alphabets
    return text

df["clean_text"] = df["text"].apply(clean_text)

# Split Data
X = df["clean_text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Save Model & Vectorizer
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("✅ Model and vectorizer saved!")
