import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "../dataset/fake_news_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("\n" + "=" * 60)
print("DATASET DIAGNOSTIC")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 2. LABEL DISTRIBUTION
# ============================================================

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLabel percentage:")
print(
    df["label"].value_counts(normalize=True) * 100
)


# ============================================================
# 3. DUPLICATE CHECK
# ============================================================

print("\nDuplicate complete rows:")
print(df.duplicated().sum())

print("\nDuplicate titles:")
print(df["title"].duplicated().sum())

print("\nDuplicate articles:")
print(df["text"].duplicated().sum())


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 5. TEXT LENGTH ANALYSIS
# ============================================================

df["title_length"] = df["title"].fillna("").astype(str).str.len()

df["text_length"] = df["text"].fillna("").astype(str).str.len()

print("\nText length by label:")

print(
    df.groupby("label")["text_length"]
    .agg(["count", "mean", "median", "min", "max"])
)


# ============================================================
# 6. CATEGORY DISTRIBUTION
# ============================================================

print("\nCategory distribution:")

print(
    pd.crosstab(
        df["category"],
        df["label"]
    )
)


# ============================================================
# 7. SOURCE DISTRIBUTION
# ============================================================

print("\nSource distribution:")

print(
    pd.crosstab(
        df["source"],
        df["label"]
    )
)


# ============================================================
# 8. TOP WORDS
# ============================================================

print("\nFinding important words...")

text = (
    df["title"].fillna("")
    + " "
    + df["text"].fillna("")
)

y = df["label"].map({
    "fake": 0,
    "real": 1
})


X_train, X_test, y_train, y_test = train_test_split(
    text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)


predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nBaseline accuracy:")
print(f"{accuracy:.4f}")


# ============================================================
# 9. MOST IMPORTANT FEATURES
# ============================================================

feature_names = vectorizer.get_feature_names_out()

coefficients = model.coef_[0]

feature_data = pd.DataFrame({
    "word": feature_names,
    "coefficient": coefficients
})


print("\nTop words associated with REAL:")

print(
    feature_data
    .sort_values("coefficient", ascending=False)
    .head(20)
    .to_string(index=False)
)


print("\nTop words associated with FAKE:")

print(
    feature_data
    .sort_values("coefficient", ascending=True)
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 11. VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="label"
)

plt.title("Fake vs Real News Distribution")

plt.xlabel("News Label")

plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    "label_distribution.png"
)

plt.close()


# ============================================================
# 12. CATEGORY VS LABEL
# ============================================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="category",
    hue="label"
)

plt.title(
    "Category Distribution by Fake/Real Label"
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "category_label_distribution.png"
)

plt.close()


# ============================================================
# 13. TEXT LENGTH
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="label",
    y="text_length"
)

plt.title(
    "Article Length by Label"
)

plt.tight_layout()

plt.savefig(
    "text_length_distribution.png"
)

plt.close()


print("\n" + "=" * 60)

print("Diagnostic completed!")

print("=" * 60)

print("\nGenerated files:")

print("label_distribution.png")
print("category_label_distribution.png")
print("text_length_distribution.png")