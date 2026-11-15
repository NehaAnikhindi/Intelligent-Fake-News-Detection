import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import clean_text


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "../dataset/fake_news_dataset.csv"

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

for column in ["title", "text", "category"]:
    df[column] = df[column].fillna("")


# ============================================================
# 3. CREATE COMBINED TEXT
# ============================================================

df["combined_text"] = (
    df["title"]
    + " "
    + df["text"]
)


# ============================================================
# 4. CLEAN TEXT
# ============================================================

print("\nCleaning text...")

df["cleaned_text"] = df["combined_text"].apply(clean_text)

df = df[df["cleaned_text"].str.strip() != ""]


# ============================================================
# 5. CONVERT LABEL
# ============================================================

df["label"] = df["label"].str.lower().str.strip()

df["target"] = df["label"].map({
    "fake": 0,
    "real": 1
})

df = df.dropna(subset=["target"])

df["target"] = df["target"].astype(int)


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

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


# ============================================================
# 7. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]),


    "Naive Bayes": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        ),

        (
            "classifier",
            MultinomialNB()
        )
    ]),


    "Linear SVM": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            LinearSVC(
                random_state=42
            )
        )
    ])
}


# ============================================================
# 8. TRAIN AND EVALUATE
# ============================================================

results = []


for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 9. DISPLAY COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 75)
print("                    MODEL COMPARISON")
print("=" * 75)

print(results_df.to_string(index=False))


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\nResults saved to:")
print("model_comparison_results.csv")