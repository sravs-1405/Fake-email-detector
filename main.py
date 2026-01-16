import pandas as pd
import re
import nltk
import joblib
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# 1. NLTK setup
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# -------------------------------
# 2. Load dataset
# -------------------------------
df = pd.read_csv("emails.csv")

# Normalize column names
df.columns = df.columns.str.lower()

# Robust column mapping
if 'message' in df.columns and 'label' in df.columns:
    df = df[['message', 'label']]

elif 'text' in df.columns and 'spam' in df.columns:
    df = df[['text', 'spam']]
    df.columns = ['message', 'label']

elif 'v2' in df.columns and 'v1' in df.columns:
    df = df[['v2', 'v1']]
    df.columns = ['message', 'label']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

else:
    raise ValueError("Dataset columns not recognized. Please check CSV.")

# Shuffle dataset for realism
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------------
# 3. Text preprocessing
# -------------------------------
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['processed_message'] = df['message'].apply(preprocess_text)

print("\nSample processed data:")
print(df.head())

# -------------------------------
# 4. Feature extraction (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['processed_message'])
y = df['label']

print("\nTF-IDF feature matrix shape:", X.shape)

# -------------------------------
# 5. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 6. Model comparison
# -------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

results = {}

print("\nMODEL COMPARISON RESULTS\n")

for name, model in models.items():
    # Wrap SVM with calibration to get probability scores
    if name == "SVM":
        model = CalibratedClassifierCV(model, cv=5)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"{name}")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))
    print("-" * 50)

# -------------------------------
# 7. Select best model
# -------------------------------
best_model_name = max(results, key=results.get)
print(f"\nBest Model Selected: {best_model_name}")

best_model = models[best_model_name]

# Wrap SVM with calibrated classifier if best
if best_model_name == "SVM":
    best_model = CalibratedClassifierCV(best_model, cv=5)

best_model.fit(X_train, y_train)

# -------------------------------
# 8. Save model & vectorizer
# -------------------------------
joblib.dump(best_model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")

# -------------------------------
# 9. Final evaluation
# -------------------------------
y_pred = best_model.predict(X_test)

print("\nFINAL MODEL EVALUATION")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 10. Custom email test with confidence
# -------------------------------
def predict_email(email_text):
    clean_text = preprocess_text(email_text)
    vector = vectorizer.transform([clean_text])

    prediction = best_model.predict(vector)[0]
    prob = best_model.predict_proba(vector)[0]
    confidence = prob[prediction] * 100

    return prediction, confidence

sample_email = input("Enter the message: ")
pred, conf = predict_email(sample_email)

print("\nCustom Email Test with Confidence:")
print("Email:", sample_email)
print("Prediction:", "SPAM" if pred == 1 else "HAM")
print(f"Confidence: {conf:.2f}%")
