# Intelligent Fake News Detection and Analysis System Using NLP and Machine Learning

A full-stack academic project that classifies submitted news articles as **FAKE** or **REAL** using text preprocessing, TF-IDF features and a trained machine-learning classifier.

## Main features

- Home page with project introduction and navigation.
- Dashboard with total, FAKE and REAL prediction counts.
- News analysis form with title, article body and category.
- Sample news input, clear form and input validation.
- Prediction result with confidence, fake probability and real probability.
- Prediction analytics chart.
- Model performance comparison loaded from `backend/model_comparison_results.csv`.
- Searchable and filterable prediction history.
- Export visible history to CSV and clear saved history.
- About, methodology, technology stack and disclaimer sections.
- Responsive purple, indigo, teal and white interface.

## Technology stack

- Frontend: React, Vite, Recharts, CSS
- Backend: Python, FastAPI, Uvicorn
- Machine learning: scikit-learn, TF-IDF, Logistic Regression
- Storage: SQLite

## Project structure

```text
fakenewsdetection/
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── model_comparison_results.csv
│   ├── models/
│   │   ├── fake_news_model.pkl
│   │   └── tfidf_vectorizer.pkl
│   └── requirements.txt
├── dataset/
│   └── fake_news_dataset.csv
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.css
│   ├── package.json
│   └── index.html
└── README.md
```

## Run on Windows

You need Python and Node.js installed.

### 1. Start the backend

Open a terminal in the project folder:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`.

Interactive API documentation: `http://127.0.0.1:8000/docs`

Keep this terminal open.

### 2. Start the frontend

Open a **second** terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite address shown in the terminal, normally `http://localhost:5173`.

Keep both terminals running while using the application.

## Test checklist

1. Open the app and confirm the API status changes to **API connected**.
2. Click **Try sample news**, then click **Analyse news**.
3. Confirm a prediction, confidence and both class probabilities appear.
4. Confirm the dashboard totals increase.
5. Confirm the article appears in Prediction History.
6. Refresh the page and confirm history remains (SQLite persistence).
7. Search history by headline and filter Fake / Real.
8. Export history to CSV and open the downloaded file.
9. Clear the form and submit empty fields; validation should appear.
10. Use Clear history and confirm the history table and dashboard counts reset.
11. Check the analytics chart and model performance comparison.
12. Resize the browser to mobile width and check the responsive layout.

## Sample test inputs

These are synthetic examples for testing the user interface and model pipeline. They are **not guaranteed labels**; the classifier decides from learned dataset patterns.

### Sample A — Science

**Title:** NASA announces a new space research mission

**Article:** NASA announced a new scientific mission designed to study distant celestial objects. Researchers will analyse the observations collected by the spacecraft and publish findings after the data has been reviewed by the mission team.

**Category:** Science

### Sample B — Suspicious claim

**Title:** Free laptop scheme requires no application

**Article:** A viral message claims that every resident will receive a free laptop immediately. It asks readers to submit personal information through an unfamiliar website and forward the message to ten contacts to claim the benefit.

**Category:** Technology

### Sample C — Sports

**Title:** National team wins international tournament

**Article:** The national team secured the tournament title after a closely contested final. Players credited their preparation and teamwork, while the coaching staff said a detailed review of the campaign would follow.

**Category:** Sports

## Model evaluation note — read before presentation

The included model comparison report currently records approximately **50–51% accuracy** for the evaluated models. This is close to chance for a balanced two-class dataset and is not strong evidence of reliable fake-news detection. Do not present this as a high-accuracy system.

Before final submission:
- Verify the dataset source, label quality and whether the text is natural, representative news.
- Check duplicates and possible train/test leakage.
- Review class balance and the confusion matrix.
- Retrain and evaluate using a stratified, held-out test set.
- Compare precision, recall and F1-score, not accuracy alone.
- Consider testing on a separate external dataset.

The metrics page displays the values from the supplied model-comparison CSV so that the evaluation is transparent. These are not live metrics calculated from user predictions.

## Important limitation

The model performs **text classification**, not independent fact verification. A FAKE/REAL prediction and its confidence are not proof that a claim is true or false. Users should verify important claims using multiple reliable sources.

## Troubleshooting

- **API offline:** Ensure the backend terminal is running and visit `http://127.0.0.1:8000/`.
- **Model file not found:** Start Uvicorn from inside the `backend` folder so the existing relative model paths resolve.
- **Frontend cannot connect:** Keep the backend on port 8000 and frontend on port 5173.
- **Missing Python package:** Activate the backend virtual environment and run `pip install -r requirements.txt`.
- **npm command not found:** Install Node.js LTS, reopen the terminal and retry.
