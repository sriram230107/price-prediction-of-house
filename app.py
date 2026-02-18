from flask import Flask, render_template, request
import numpy as np
import joblib
import sqlite3
import os

print("App file location:", __file__)

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
app = Flask(__name__, template_folder=template_dir)

# Load models
basic_model = joblib.load("basic_model.pkl")
advanced_model = joblib.load("advanced_model.pkl")


# =========================
# CREATE DATABASE TABLE
# =========================
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bedrooms REAL,
            bathrooms REAL,
            living_area REAL,
            floors REAL,
            grade REAL,
            built_year REAL,
            price REAL,
            model TEXT,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

init_db()


# =========================
# HOME ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    price = None

    if request.method == "POST":

        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])
        living_area = float(request.form["living_area"])
        use_advanced = request.form.get("use_advanced")

        if use_advanced:
            floors = float(request.form.get("floors") or 0)
            grade = float(request.form.get("grade") or 0)
            built_year = float(request.form.get("built_year") or 0)

            features = np.array([[bedrooms, bathrooms, living_area,
                                  floors, grade, built_year]])

            prediction = advanced_model.predict(features)

            model_used = "Advanced"

        else:
            features = np.array([[bedrooms, bathrooms, living_area]])
            prediction = basic_model.predict(features)

            floors = grade = built_year = None
            model_used = "Basic"

        price = round(prediction[0], 2)

        # SAVE TO DATABASE
        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions
            (bedrooms, bathrooms, living_area, floors, grade, built_year, price, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (bedrooms, bathrooms, living_area, floors, grade, built_year, price, model_used))

        conn.commit()
        conn.close()

    return render_template("index.html", price=price)


# =========================
# HISTORY ROUTE
# =========================
@app.route("/history")
def history():
    conn = sqlite3.connect("predictions.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()

    conn.close()

    return render_template("history.html", rows=rows)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
