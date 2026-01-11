import pandas as pd
import hopsworks
import joblib
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime
import os

def get_weather_forecast():
    print("📡 Connecting to Open-Meteo Forecast API (Future Data)...")
    
    # 配置 API (注意: 这里是 Forecast API，不是 Archive)
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # 1. 请求未来 7 天的天气预报
    params = {
        "latitude": 63.399,  # Åre
        "longitude": 13.082,
        "daily": ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "snowfall_sum"],
        "timezone": "Europe/Berlin",
        "forecast_days": 7
    }
    
    url = "https://api.open-meteo.com/v1/forecast"
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # 2. 处理数据
    daily = response.Daily()
    daily_data = {
        "date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )
    }
    
    # 注意：这里的变量名必须和训练时的 Feature Group 列名完全一致！
    # 训练时我们用了: temperature_max, precipitation, wind_gusts, snowfall_sum
    daily_data["temperature_max"] = daily.Variables(0).ValuesAsNumpy()
    daily_data["precipitation"] = daily.Variables(1).ValuesAsNumpy()
    daily_data["wind_gusts"] = daily.Variables(2).ValuesAsNumpy()
    daily_data["snowfall_sum"] = daily.Variables(3).ValuesAsNumpy()
    
    df = pd.DataFrame(data = daily_data)
    
    # 转换日期格式为字符串 (用于展示)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    
    return df

def run_inference():
    # 1. 登录 Hopsworks
    print("🔐 Logging into Hopsworks...")
    project = hopsworks.login()
    
    # 2. 获取模型
    print("📥 Downloading model from Registry...")
    mr = project.get_model_registry()
    model = mr.get_model(name="ski_depth_model", version=1)
    model_dir = model.download()
    
    # 加载本地模型文件
    # 注意：根据你之前训练脚本的保存名，这里应该是 sklearn_ski_model.pkl
    model_path = os.path.join(model_dir, "sklearn_ski_model.pkl")
    trained_model = joblib.load(model_path)
    
    # 3. 获取未来天气预报
    df_forecast = get_weather_forecast()
    print(f"🌦️ Fetched 7-day forecast for Åre. Rows: {len(df_forecast)}")
    
    # 4. 准备特征数据 (去掉日期列，只留模型需要的 4 个特征)
    # 必须和训练时的 X_train 列顺序一致
    features = df_forecast[['temperature_max', 'precipitation', 'wind_gusts', 'snowfall_sum']]
    
    # 5. 进行预测
    print("🔮 Predicting snow depth...")
    predictions = trained_model.predict(features)
    
    # 6. 展示结果
    print("\n⛷️  Ski Forecast for Åre (Next 7 Days):")
    print("-" * 50)
    for date, snow_depth, temp in zip(df_forecast['date_str'], predictions, df_forecast['temperature_max']):
        # 处理负数预测 (模型可能会预测出 -0.1cm，修正为 0)
        snow_depth = max(0, snow_depth)
        
        condition = "❄️ Good Snow" if snow_depth > 10 else "🌱 No Snow"
        print(f"📅 {date} | Temp: {temp:5.1f}°C | Snow Depth: {snow_depth:5.1f} cm | {condition}")
    print("-" * 50)

if __name__ == "__main__":
    run_inference()