import gradio as gr
import hopsworks
import joblib
import pandas as pd
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry
from huggingface_hub import InferenceClient

# ==========================================
# 🧠 Model 1: 预测管道 (获取硬数据)
# ==========================================
def get_weather_forecast():
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    params = {
        "latitude": 63.399, "longitude": 13.082, # Åre
        "daily": ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "snowfall_sum"],
        "timezone": "Europe/Berlin", "forecast_days": 7
    }
    
    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    response = responses[0]
    daily = response.Daily()
    
    df = pd.DataFrame({
        "date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        ),
        "temperature_max": daily.Variables(0).ValuesAsNumpy(),
        "precipitation": daily.Variables(1).ValuesAsNumpy(),
        "wind_gusts": daily.Variables(2).ValuesAsNumpy(),
        "snowfall_sum": daily.Variables(3).ValuesAsNumpy(),
    })
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    return df

def get_prediction_data():
    print("🤖 Model 1 working: Connecting to Hopsworks...")
    try:
        # 这里的 login 会自动读取 Secrets 里的 HOPSWORKS_API_KEY
        project = hopsworks.login(project="zeyashen")
        mr = project.get_model_registry()
        model = mr.get_model(name="ski_depth_model", version=1)
        model_dir = model.download()
        
        model_path = os.path.join(model_dir, "sklearn_ski_model.pkl")
        trained_model = joblib.load(model_path)
        
        df = get_weather_forecast()
        features = df[['temperature_max', 'precipitation', 'wind_gusts', 'snowfall_sum']]
        
        preds = trained_model.predict(features)
        preds = [max(0, p) for p in preds] # 修正负数
        
        summary = ""
        for date, snow, temp in zip(df['date_str'], preds, df['temperature_max']):
            summary += f"- {date}: Temp {temp:.1f}°C, Predicted Snow Depth {snow:.1f}cm\n"
        
        return summary
    except Exception as e:
        print(f"Error: {e}")
        return "Error fetching data. Please check logs."

print("⏳ Initializing: Fetching latest data and model...")
CACHE_FORECAST = get_prediction_data()
print("✅ Data ready!")

# ==========================================
# 🗣️ Model 2: Hugging Face LLM (创意对话)
# ==========================================
def chatbot_response(message, history):
    token = os.environ.get("HF_TOKEN")
    if not token:
        yield "⚠️ Error: HF_TOKEN is missing. Please add it in Settings -> Secrets."
        return

    # # 🔄 修改点：换成了更稳定的 Microsoft Phi-3.5 模型
    # client = InferenceClient("microsoft/Phi-3.5-mini-instruct", token=token)

    client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=token)
    
    system_prompt = f"""
    You are 'SnowBot', a funny ski instructor in Åre, Sweden.
    REAL forecast data:
    {CACHE_FORECAST}
    Rules: Short answer. Use emojis. Be sarcastic if no snow, excited if deep snow.
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    try:
        partial_message = ""
        for token in client.chat_completion(messages, max_tokens=500, stream=True):
            if token.choices[0].delta.content:
                partial_message += token.choices[0].delta.content
                yield partial_message
    except Exception as e:
        # 这里会把详细错误打印在对话框里，方便调试
        yield f"LLM Error: {str(e)}"

# ==========================================
# 🎨 Gradio 界面
# ==========================================
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="🎿 Åre Ski Forecast Bot",
    description="System 2: XGBoost Prediction + Zephyr-7B LLM.",
    examples=[
        "Is it worth going skiing tomorrow?",
        "How is the snow on the weekend?",
    ],
    # 关键修复：关闭示例缓存，防止启动时崩溃
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()