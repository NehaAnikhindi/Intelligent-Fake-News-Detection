import sqlite3

DATABASE = "news_predictions.db"


def create_table():
    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            category TEXT,
            prediction TEXT,
            confidence REAL,
            fake_probability REAL,
            real_probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(
    title,
    text,
    category,
    prediction,
    confidence,
    fake_probability,
    real_probability
):
    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (
            title,
            text,
            category,
            prediction,
            confidence,
            fake_probability,
            real_probability
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        title,
        text,
        category,
        prediction,
        confidence,
        fake_probability,
        real_probability
    ))

    connection.commit()
    connection.close()


def get_predictions():
    connection = sqlite3.connect(DATABASE)

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


def delete_predictions():
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM predictions")
    connection.commit()
    connection.close()
