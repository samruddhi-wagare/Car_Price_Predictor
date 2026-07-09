# 🚗 Car Price Predictor

A Machine Learning-powered web application designed to predict the market value of used cars. By leveraging regression algorithms, this system evaluates key automobile features—such as manufacturing year, driven mileage, fuel type, brand, and vehicle model—to calculate real-time, data-driven pricing estimates.

---

## 🛠️ Key Features

* **Data Cleanup & Preprocessing:** Handles missing attributes, normalizes skewed target variables, and uses modern categorical encoding (e.g., One-Hot Encoding).
* **Predictive Intelligence:** Built using supervised machine learning regression algorithms to identify hidden pricing patterns across multiple vehicle generations.
* **Intuitive Web Interface:** Offers a clean, minimal user experience where users can dynamically select vehicle specifications and receive instant predictions.
* **Responsive Input Fields:** Implements dynamic lookups to filter specific models mapped exactly to their parental manufacturing companies.

---

## 💻 Tech Stack & Libraries

* **Language:** Python
* **Data Analysis & Modeling:** Pandas, NumPy, Scikit-learn
* **Data Visualization:** Matplotlib, Seaborn
* **Web Framework:** Flask / Streamlit
* **Deployment:** Render / Heroku / Streamlit Community Cloud

---

## 📊 Workflow Pipeline
[Raw Dataset] ➔ [Data Cleaning & EDA] ➔ [Feature Engineering]
↓
[User Interface] ← [Serialized Model (.pkl)] ← [Model Training & Evaluation]


