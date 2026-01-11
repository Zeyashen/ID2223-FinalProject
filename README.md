# ID2223-FinalProject
# 🎿 Åre Ski Snow Depth Predictor (ID2223 Final Project)

<div align="center">

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Hopsworks](https://img.shields.io/badge/Hopsworks-Feature%20Store-green)](https://www.hopsworks.ai/)

**A Serverless ML System that predicts snow depth in Åre, Sweden, and chats about it using an LLM.**

</div>

## 📖 Project Overview

This project implements a complete **MLOps pipeline** to predict ski conditions (snow depth) for the upcoming week in Åre, Sweden. It utilizes a **System 2** architecture, combining a traditional Machine Learning model (XGBoost/Sklearn) for factual predictions with a Large Language Model (Qwen-2.5) for natural language interaction.

### 🌟 Key Features
* **Historical Data:** 20 years of weather data sourced from Open-Meteo API.
* **Feature Store:** Hopsworks for managing features and training datasets.
* **Model Registry:** Versioning and storing the trained regression model.
* **Daily Inference:** Fetches live weather forecasts to predict future snow depth.
* **LLM Integration:** Uses `Qwen/Qwen2.5-7B-Instruct` to interpret the data and act as a sarcastic ski instructor via a Chatbot UI.

---

## 🏗️ Architecture (System 2 Design)

The system is split into three independent pipelines (Feature, Training, Inference) to ensure modularity and scalability.

```mermaid
graph LR
    A[Open-Meteo API] -->|Historical Data| B(Feature Pipeline)
    B -->|Feature Group| C{Hopsworks Feature Store}
    C -->|Training Data| D(Training Pipeline)
    D -->|Trained Model| E{Model Registry}
    
    F[Open-Meteo Forecast] -->|Future Weather| G(Hugging Face Space)
    E -->|Download Model| G
    
    G -->|Facts: Snow Depth| H[LLM: Qwen-2.5]
    H -->|Natural Language Response| I[User Interface / Gradio]