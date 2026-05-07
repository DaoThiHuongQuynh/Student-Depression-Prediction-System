import os
import sys
import json
import numpy as np

# --- CẤU HÌNH MÔI TRƯỜNG (Giữ nguyên như cũ vì nó đã chạy được) ---
PYTHON_PATH = r"F:\xlabienso\pythonProject1\.venv\Scripts\python.exe"
os.environ['PYSPARK_PYTHON'] = PYTHON_PATH
os.environ['PYSPARK_DRIVER_PYTHON'] = PYTHON_PATH
os.environ['HADOOP_HOME'] = r"E:\hadoop"
sys.path.append(r"E:\hadoop\bin")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline


def train_and_export():
    print("⏳ Đang khởi tạo Spark...")
    spark = SparkSession.builder \
        .appName("TrainAndExport") \
        .master("local[*]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()

    # 1. ĐỌC DỮ LIỆU
    csv_file = "survey_expert_cleaned.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Thiếu file: {csv_file}")
        return

    print("📂 Đang đọc và xử lý dữ liệu...")
    df = spark.read.csv(csv_file, sep=',', header=True, inferSchema=True)
    if len(df.columns) <= 1:
        df = spark.read.csv(csv_file, sep=';', header=True, inferSchema=True)

    # Xử lý dữ liệu (Giống hệt các bước trước)
    if "Gender" in df.columns:
        df = df.withColumn("Gender", when(col("Gender") == "Male", 1).otherwise(0))
    if "Family History of Mental Illness" in df.columns:
        df = df.withColumn("Family History of Mental Illness",
                           when(col("Family History of Mental Illness") == "Yes", 1).otherwise(0))
    if "Sleep Duration" in df.columns:
        df = df.withColumn("Sleep Duration",
                           when(col("Sleep Duration") == "Less than 5 hours", 0)
                           .when(col("Sleep Duration") == "5-6 hours", 1)
                           .when(col("Sleep Duration") == "7-8 hours", 2)
                           .otherwise(3)
                           )

    input_cols = ["Age", "CGPA", "Academic Pressure", "Study Satisfaction", "Financial Stress", "Sleep Duration",
                  "Gender", "Family History of Mental Illness"]
    existing_cols = [c for c in input_cols if c in df.columns]

    for column in existing_cols:
        mean_val = df.select(mean(col(column))).collect()[0][0]
        df = df.fillna(mean_val, subset=[column])

    # 2. HUẤN LUYỆN MODEL
    print("🧠 Đang huấn luyện (Training)...")
    assembler = VectorAssembler(inputCols=existing_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    lr = LogisticRegression(featuresCol="features", labelCol="Depression")

    pipeline = Pipeline(stages=[assembler, scaler, lr])
    model = pipeline.fit(df)
    print("✅ Huấn luyện xong!")

    # 3. TRÍCH XUẤT THAM SỐ NGAY LẬP TỨC (Lấy từ RAM ra luôn, không cần Load lại)
    print("📥 Đang trích xuất tham số ra JSON...")

    # Lấy Model con bên trong Pipeline
    # Stages: [0]Assembler, [1]ScalerModel, [2]LogisticRegressionModel

    scaler_model = model.stages[1]
    lr_model = model.stages[2]

    # Chuyển đổi sang List của Python để lưu JSON
    means = scaler_model.mean.toArray().tolist()
    stds = scaler_model.std.toArray().tolist()
    coefficients = lr_model.coefficients.toArray().tolist()
    intercept = lr_model.intercept

    data_json = {
        "means": means,
        "stds": stds,
        "coefficients": coefficients,
        "intercept": intercept
    }

    # Lưu file JSON
    with open("model_params.json", "w") as f:
        json.dump(data_json, f)

    print("-" * 30)
    print("✅ THÀNH CÔNG! Đã tạo file 'model_params.json'")
    print("-" * 30)

    spark.stop()


if __name__ == "__main__":
    train_and_export()