import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing import clean_text


# =========================================================
# 1. LOAD DATASET
# =========================================================

DATASET_PATH = "../dataset/fake_news_dataset.csv"

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# 2. DISPLAY BASIC INFORMATION
# =========================================================

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
print(df["label"].value_counts())


# =========================================================
# 3. HANDLE MISSING VALUES
# =========================================================

text_columns = ["title", "text", "category"]

for column in text_columns:
    df[column] = df[column].fillna("")


# =========================================================
# 4. COMBINE TEXT FEATURES
# =========================================================

print("\nCombining title, article text and category...")

df["combined_text"] = (
    df["title"] + " "
    + df["text"] + " "
    + df["category"]
)


# =========================================================
# 5. TEXT PREPROCESSING
# =========================================================

print("\nCleaning text...")

df["cleaned_text"] = df["combined_text"].apply(clean_text)

print("Text preprocessing completed.")


# =========================================================
# 6. REMOVE EMPTY RECORDS
# =========================================================

df = df[df["cleaned_text"].str.strip() != ""]

print("\nDataset after cleaning:", df.shape)


# =========================================================
# 7. CONVERT LABELS
# =========================================================

df["label"] = df["label"].str.lower().str.strip()

label_mapping = {
    "fake": 0,
    "real": 1
}

df["target"] = df["label"].map(label_mapping)

# Remove unexpected labels
df = df.dropna(subset=["target"])

df["target"] = df["target"].astype(int)


# =========================================================
# 8. TRAIN / TEST SPLIT
# =========================================================

X = df["cleaned_text"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 9. TF-IDF
# =========================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# =========================================================
# 10. TRAIN LOGISTIC REGRESSION
# =========================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# =========================================================
# 11. PREDICTION
# =========================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test_tfidf)


# =========================================================
# 12. EVALUATION
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n====================================")
print("       MODEL PERFORMANCE")
print("====================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake", "Real"],
        zero_division=0
    )
)


# =========================================================
# 13. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# 14. SAVE MODEL
# =========================================================

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "fake_news_model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

print("\n====================================")
print("Models saved successfully!")
print("====================================")

print("\nSaved files:")
print("models/fake_news_model.pkl")
print("models/tfidf_vectorizer.pkl")