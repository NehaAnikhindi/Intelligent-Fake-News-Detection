# Fake News Detection Project - Setup & Testing Notes

## Project structure
- `frontend/`: React + Vite user interface
- `backend/`: FastAPI API, NLP preprocessing, trained TF-IDF + Logistic Regression model
- `dataset/`: CSV training dataset
- `backend/models/`: saved model and vectorizer

## Important
This system is a machine-learning text classifier, not a live fact-checking service. Its FAKE/REAL output is a prediction based on patterns learned from the training dataset. Do not present the score as proof that a story is true or false.

## Run locally (Windows)
1. Open two terminals in VS Code.
2. Backend:
   - `cd fakenewsdetection/backend`
   - Create environment: `py -3.12 -m venv .venv`
   - Activate: `.venv\\Scripts\\activate`
   - Install: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --reload`
3. Frontend:
   - `cd fakenewsdetection/frontend`
   - `npm install`
   - `npm run dev`
4. Open the Vite address shown in the terminal (usually `http://localhost:5173`).
5. API documentation: `http://127.0.0.1:8000/docs`.

If NLTK resources are not installed, run once in the backend environment:
`python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"`

## Manual test inputs
Use the sample inputs in the accompanying project review response. Results can vary with the dataset and trained model; they are not guaranteed labels.

## Before final submission
- Verify dataset source, licence, number of rows, label balance and missing values.
- Report test-set accuracy, precision, recall, F1-score and confusion matrix from a held-out test set.
- Check that model class order is used correctly when mapping probabilities to FAKE and REAL.
- Add input validation, clear loading/error states, history search/filter and a clear-history action (with confirmation).
- Add a disclaimer and privacy note. Avoid storing full article text unless necessary; provide a retention/deletion policy.
- Add automated backend tests and frontend build checks.
- Remove virtual environments, `node_modules`, caches and local database files from GitHub submissions; include requirements and lockfiles instead.
