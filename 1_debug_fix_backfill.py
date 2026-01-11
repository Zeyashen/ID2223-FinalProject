import pandas as pd
import hopsworks
import openmeteo_requests
import requests_cache
from retry_requests import retry

def get_historical_snow_data():
    print("📡 Connecting to Open-Meteo Archive API...")
    
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # 获取最近2年的数据
    params = {
        "latitude": 63.399,
        "longitude": 13.082,
        "start_date": "2005-01-01", # 扩大一点范围
        "end_date": "2025-01-01",
        "daily": ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "snowfall_sum"],
        "hourly": ["snow_depth"],
        "timezone": "Europe/Berlin"
    }
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # --- 1. 处理 Daily 数据 ---
    daily = response.Daily()
    daily_data = {
        "date_obj": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )
    }
    daily_data["temperature_max"] = daily.Variables(0).ValuesAsNumpy()
    daily_data["precipitation"] = daily.Variables(1).ValuesAsNumpy()
    daily_data["wind_gusts"] = daily.Variables(2).ValuesAsNumpy()
    daily_data["snowfall_sum"] = daily.Variables(3).ValuesAsNumpy()
    
    df_daily = pd.DataFrame(data = daily_data)
    # 创建一个纯净的 join key
    df_daily['join_date'] = df_daily['date_obj'].dt.strftime('%Y-%m-%d')
    
    print(f"✅ Daily data fetched. Rows: {len(df_daily)}")

    # --- 2. 处理 Hourly 数据 ---
    hourly = response.Hourly()
    hourly_data = {
        "date_time": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )
    }
    hourly_data["snow_depth"] = hourly.Variables(0).ValuesAsNumpy()
    df_hourly = pd.DataFrame(data = hourly_data)
    
    # 聚合到天：把小时转成天字符串
    df_hourly['join_date'] = df_hourly['date_time'].dt.strftime('%Y-%m-%d')
    # 取每天最大的雪深
    df_snow_daily = df_hourly.groupby('join_date')['snow_depth'].max().reset_index()
    
    print(f"✅ Hourly aggregated. Rows: {len(df_snow_daily)}")

    # --- 3. 合并 (使用字符串 key，最稳妥) ---
    df = pd.merge(df_daily, df_snow_daily, on='join_date', how='inner')
    print(f"🔗 Merged data. Rows: {len(df)}")

    # --- 4. 清洗 ---
    df = df.dropna()
    print(f"🧹 After dropping NaNs. Rows: {len(df)}")
    
    if len(df) == 0:
        raise ValueError("❌ Error: Dataframe is empty! Check API parameters or merge logic.")

    # --- 5. 格式化最终列 ---
    # 恢复 date 为时间戳 (Event Time)
    df['date'] = pd.to_datetime(df['join_date'])
    # 创建 date_str (Primary Key)
    df['date_str'] = df['join_date'].astype("string")
    
    # 删除临时列
    df = df.drop(columns=['date_obj', 'join_date'])
    
    # 重新排列列顺序，好看一点
    cols = ['date', 'date_str', 'temperature_max', 'precipitation', 'wind_gusts', 'snowfall_sum', 'snow_depth']
    df = df[cols]
    
    return df

def run_job():
    # 1. 获取数据
    try:
        df = get_historical_snow_data()
        print(f"📊 Final Dataset Preview:\n{df.head()}")
    except Exception as e:
        print(e)
        return

    # 2. 登录 Hopsworks
    print("🔐 Logging into Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    # 3. 删除旧的 Version 2 (如果有)
    print("🧹 Checking for old empty Feature Group...")
    try:
        old_fg = fs.get_feature_group(name="ski_weather_data", version=2)
        old_fg.delete()
        print("🗑️ Deleted old empty Feature Group version 2.")
    except:
        print("ℹ️ No old version 2 found.")

    # 4. 创建并上传
    print("🚀 Uploading to Feature Store (Version 2)...")
    
    ski_fg = fs.get_or_create_feature_group(
        name="ski_weather_data",
        version=3,
        primary_key=["date_str"], 
        event_time="date",
        description="Daily weather and aggregated snow depth for Are",
        online_enabled=True
    )

    ski_fg.insert(df)
    print("🎉 Success! Data successfully backfilled.")
    print("⏳ Please wait 1-2 minutes for the data to be indexed before running the training script.")

if __name__ == "__main__":
    run_job()