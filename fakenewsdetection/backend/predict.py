import os
import joblib

from preprocessing import clean_text


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "fake_news_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    "models",
    "tfidf_vectorizer.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ML model...")

model = joblib.load(MODEL_PATH)

vectorizer = joblib.load(VECTORIZER_PATH)

print("Model loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_news(title, text, category=""):

    # Combine input fields
    combined_text = (
        str(title)
        + " "
        + str(text)
        + " "
        + str(category)
    )

    # Clean text
    cleaned_text = clean_text(
        combined_text
    )

    # Convert text into TF-IDF
    text_vector = vectorizer.transform(
        [cleaned_text]
    )

    # Prediction
    prediction = model.predict(
        text_vector
    )[0]

    # Probability
    probabilities = model.predict_proba(
        text_vector
    )[0]

    # Map probabilities using the model's learned class labels rather than
    # assuming a fixed ordering of predict_proba columns.
    class_probabilities = {
        int(class_label): float(probability)
        for class_label, probability in zip(model.classes_, probabilities)
    }
    fake_probability = class_probabilities.get(0, 0.0)
    real_probability = class_probabilities.get(1, 0.0)

    # Label
    if prediction == 0:
        label = "FAKE"
        confidence = fake_probability
    else:
        label = "REAL"
        confidence = real_probability

    return {
        "prediction": label,
        "confidence": round(
            float(confidence) * 100,
            2
        ),
        "fake_probability": round(
            float(fake_probability) * 100,
            2
        ),
        "real_probability": round(
            float(real_probability) * 100,
            2
        )
    }


# ============================================================
# TEST PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("        FAKE NEWS DETECTION SYSTEM")
    print("=" * 60)

    title = input(
        "\nEnter news title: "
    )

    text = input(
        "\nEnter news article: "
    )

    category = input(
        "\nEnter category (optional): "
    )

    result = predict_news(
        title,
        text,
        category
    )

    print("\n")
    print("=" * 60)
    print("                RESULT")
    print("=" * 60)

    print(
        f"\nPrediction       : "
        f"{result['prediction']}"
    )

    print(
        f"Confidence       : "
        f"{result['confidence']}%"
    )

    print(
        f"Fake Probability : "
        f"{result['fake_probability']}%"
    )

    print(
        f"Real Probability : "
        f"{result['real_probability']}%"
    )

    print("\n" + "=" * 60)