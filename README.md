# 🎿 Åre Ski Snow Depth Predictor (ID2223 Final Project)

<div align="center">

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE_NAME)
[![Hopsworks](https://img.shields.io/badge/Hopsworks-Feature%20Store-green)](https://www.hopsworks.ai/)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-orange)](https://scikit-learn.org/)

**A Serverless "System 2" ML application that predicts snow depth in Åre, Sweden, and provides a sarcastic ski forecast via an LLM.**

</div>

---

## 📋 Project Proposal & Requirements
*(Addressing the specific requirements for ID2223 Final Project)*

### 0. Project Name and Group Members
* **Project Name:** Åre Ski Snow Depth Predictor
* **Group Members:** Zeya Shen

### 1. Dynamic Data Sources
We utilize **dynamic, real-time APIs** instead of static datasets (like Kaggle).
* **Training Data:** Fetched from **Open-Meteo Historical Weather API**. We ingest 20 years of daily weather history (2005-2025) including temperature, precipitation, wind speed, and snow depth.
* **Inference Data:** The application fetches live 7-day weather forecasts from the **Open-Meteo Forecast API** every time a user requests a prediction.

### 2. The Prediction Problem
We are solving a **Regression Problem** to predict the **Snow Depth (cm)** for the upcoming week in Åre.
* **Target:** `snow_depth` (Daily Maximum in cm).
* **Features:** `temperature_2m_max`, `precipitation_sum`, `wind_speed_10m_max`, `snowfall_sum`.
* **Model:** `HistGradientBoostingRegressor` (Scikit-Learn).

### 3. The UI (System 2 Design)
We provide an interactive **Chatbot Interface** hosted on **Hugging Face Spaces**.
The system employs a "System 2" architecture:
* **System 1 (Analytical):** The XGBoost/Sklearn model calculates the exact snow depth (e.g., "45 cm") based on weather forecasts.
* **System 2 (Generative):** A Large Language Model (**Qwen-2.5-7B**) interprets this data. It acts as a "Sarcastic Ski Instructor," translating raw numbers into fun, actionable advice (e.g., *"Only 2cm of snow? Go drink hot chocolate instead."*).

### 4. Overview of Technologies
* **Feature Store:** Hopsworks (for feature engineering and storage).
* **Machine Learning:** Scikit-Learn 1.3.2 (Model Training).
* **LLM Inference:** Hugging Face Inference API (`Qwen/Qwen2.5-7B-Instruct`).
* **Web UI:** Gradio.
* **Deployment:** Hugging Face Spaces (Dockerized environment).

---

## 🏗️ System Architecture

The project is structured into three decoupled pipelines following MLOps best practices:

```mermaid
graph LR
    subgraph Feature Pipeline
        A[Open-Meteo Archive] -->|Fetch 20yrs Data| B(1_backfill_feature_group.py)
        B -->|Write| C{Hopsworks Feature Store}
    end

    subgraph Training Pipeline
        C -->|Read Feature View| D(2_training_pipeline.py)
        D -->|Train HistGradientBoosting| E(Model Registry)
    end

    subgraph Inference Pipeline
        F[Open-Meteo Forecast] -->|Fetch 7-day Forecast| G(Hugging Face Space / app.py)
        E -->|Download Model| G
        G -->|Predict Snow Depth| H[Analytical Result]
        H -->|Prompt Engineering| I[LLM: Qwen-2.5]
        I -->|Natural Language Response| J[User Chat UI]
    end