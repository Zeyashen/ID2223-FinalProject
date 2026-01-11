import gradio as gr
import hopsworks
import joblib
import pandas as pd
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry
from huggingface_hub import InferenceClient # <--- 替换了 OpenAI

# ==========================================
# 🔑 配置区域
# ==========================================
# 确保你的环境中配置了 HF_TOKEN
# 或者临时填在这里 (提交到 GitHub 前记得删掉！)
if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = "bcjidsfbhewqilfewuyi768321bfhewqvbhkl290427bdkscdskvs"

# ==========================================
# 🧠 Model 1: 预测管道 (获取硬数据 - 不变)
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
    """
    运行模型预测，返回一个包含未来7天数据的字符串摘要
    """
    print("🤖 Model 1 working: Connecting to Hopsworks...")
    try:
        project = hopsworks.login()
        mr = project.get_model_registry()
        model = mr.get_model(name="ski_depth_model", version=1)
        model_dir = model.download()
        
        model_path = os.path.join(model_dir, "sklearn_ski_model.pkl")
        trained_model = joblib.load(model_path)
        
        df = get_weather_forecast()
        features = df[['temperature_max', 'precipitation', 'wind_gusts', 'snowfall_sum']]
        
        preds = trained_model.predict(features)
        preds = [max(0, p) for p in preds] # 修正负数
        
        # 构建数据摘要
        summary = ""
        for date, snow, temp in zip(df['date_str'], preds, df['temperature_max']):
            summary += f"- {date}: Temp {temp:.1f}°C, Predicted Snow Depth {snow:.1f}cm\n"
        
        return summary
    except Exception as e:
        return f"Error fetching predictions: {str(e)}"

# 启动时先获取一次数据，存入缓存
print("⏳ Initializing: Fetching latest data and model...")
CACHE_FORECAST = get_prediction_data()
print("✅ Data ready!")

# ==========================================
# 🗣️ Model 2: Hugging Face LLM (创意对话)
# ==========================================
def chatbot_response(message, history):
    """
    使用 Hugging Face Inference API 进行对话
    """
    
    # 1. 初始化客户端
    # 我们使用 Zephyr-7b-beta，它是 Lab 2 中常用的 Mistral 的优化版，对话能力很强
    # 你也可以换成 "mistralai/Mistral-7B-Instruct-v0.3"
    client = InferenceClient(
        "HuggingFaceH4/zephyr-7b-beta", 
        token=os.environ["HF_TOKEN"]
    )
    
    # 2. 定义角色 (System Prompt)
    system_prompt = f"""
    You are 'SnowBot', a funny and slightly sarcastic ski instructor in Åre, Sweden.
    
    Here is the REAL forecast (from our ML model) for the next 7 days:
    {CACHE_FORECAST}
    
    Rules:
    1. Answer based on the data above.
    2. If snow < 10cm, be pessimistic and sarcastic.
    3. If snow > 30cm, be super excited!
    4. Keep answers short (under 3 sentences).
    5. Use emojis!
    """

    # 3. 构建 Prompt (格式化为 Chat 结构)
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    
    # 添加历史记录
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    messages.append({"role": "user", "content": message})

    # 4. 生成回复
    # stream=True 让回复像打字机一样出来，体验更好
    partial_message = ""
    for token in client.chat_completion(messages, max_tokens=500, stream=True):
        if token.choices[0].delta.content:
            partial_message += token.choices[0].delta.content
            yield partial_message

# ==========================================
# 🎨 Gradio 界面
# ==========================================
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="🎿 Åre Ski Forecast Bot (System 2 Design)",
    description="I use a Prediction Model (XGBoost/Sklearn) for facts, and an LLM (Zephyr-7B) for personality.",
    examples=[
        "Is it worth going skiing tomorrow?",
        "How is the snow on the weekend?",
        "Should I bring my rock skis?",
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()