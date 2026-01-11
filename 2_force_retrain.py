import hopsworks
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_model():
    print("🔐 Logging into Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    # --- 关键步骤：清理旧的 Feature View ---
    fv_name = "ski_weather_data"
    fv_version = 3
    
    print(f"🧹 Cleaning up: Attempting to delete Feature View {fv_name} version {fv_version}...")
    try:
        # 先尝试获取，如果获取到了，直接删除
        fv = fs.get_feature_view(fv_name, version=fv_version)
        fv.delete()
        print("🗑️ Successfully deleted old/corrupted Feature View.")
    except Exception as e:
        print("ℹ️ No old Feature View found (or delete failed), continuing...")

    # --- 重新创建 ---
    print("🆕 Creating a fresh Feature View...")
    
    # 1. 获取 Feature Group (Version 2)
    # 确保这里是 2，因为这是我们 backfill 数据的版本
    ski_fg = fs.get_feature_group(name="ski_weather_data", version=2)
    
    # 2. 创建 Feature View
    query = ski_fg.select_all()
    
    feature_view = fs.create_feature_view(
        name=fv_name,
        version=fv_version,
        description="Read weather and predict snow depth",
        labels=["snow_depth"],
        query=query
    )
    
    print(f"✅ Feature View created. Type: {type(feature_view)}")
    
    if feature_view is None:
        print("❌ CRITICAL: Create returned None! Stopping.")
        return

    # 3. 划分训练集和测试集
    print("✂️ Splitting data into Train/Test sets...")
    X_train, X_test, y_train, y_test = feature_view.train_test_split(
        test_size=0.2,
        description="Ski weather forecast dataset split"
    )

    # --- 数据清理 ---
    drop_cols = ["date", "date_str", "snow_depth"]
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
    
    print(f"Training features: {X_train.columns.tolist()}")

    # 4. 训练模型
    print("🧠 Training HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.1, 
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5. 评估模型
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    
    print(f"✅ Model Evaluation Results:")
    print(f"   RMSE: {rmse:.2f} cm")
    print(f"   R2 Score: {r2:.2f}")

    # 6. 保存并注册模型
    print("💾 Saving model to Hopsworks Model Registry...")
    
    model_dir = "ski_model"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    joblib.dump(model, model_dir + "/sklearn_ski_model.pkl")

    mr = project.get_model_registry()
    
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    ski_model = mr.python.create_model(
        name="ski_depth_model", 
        metrics={"rmse": rmse, "r2": r2},
        model_schema=model_schema,
        input_example=X_train.sample(), 
        description="Predicts snow depth (cm) based on weather forecast (Sklearn)"
    )
    
    ski_model.save(model_dir)
    print("🎉 Model successfully registered!")

if __name__ == "__main__":
    train_model()