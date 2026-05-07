import os
import sys
import json
import numpy as np

# Cấu hình môi trường (Giống train_local.py đã chạy thành công)
PYTHON_PATH = r"F:\xlabienso\pythonProject1\.venv\Scripts\python.exe"
os.environ['PYSPARK_PYTHON'] = PYTHON_PATH
os.environ['PYSPARK_DRIVER_PYTHON'] = PYTHON_PATH
os.environ['HADOOP_HOME'] = r"E:\hadoop"
sys.path.append(r"E:\hadoop\bin")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


def export():
    print("⏳ Đang khởi động Spark để lấy tham số...")
    spark = SparkSession.builder \
        .appName("ExportParams") \
        .master("local[*]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()

    model_path = "spark_depression_model"
    if not os.path.exists(model_path):
        print("❌ Lỗi: Không thấy thư mục model. Hãy chạy train_local.py trước!")
        return

    print("📂 Đang đọc Model...")
    pipeline_model = PipelineModel.load(model_path)

    # --- TRÍCH XUẤT THAM SỐ TOÁN HỌC ---
    # Stage 1: StandardScaler (Lấy Mean và Std để chuẩn hóa dữ liệu)
    scaler = pipeline_model.stages[1]
    means = scaler.mean.toArray().tolist()
    stds = scaler.std.toArray().tolist()

    # Stage 2: LogisticRegression (Lấy hệ số W và chặn b)
    lr = pipeline_model.stages[2]
    coefficients = lr.coefficients.toArray().tolist()
    intercept = lr.intercept

    # Lưu vào file JSON
    data = {
        "means": means,
        "stds": stds,
        "coefficients": coefficients,
        "intercept": intercept
    }

    with open("model_params.json", "w") as f:
        json.dump(data, f)

    print("✅ ĐÃ XUẤT THÀNH CÔNG FILE: model_params.json")
    spark.stop()


if __name__ == "__main__":
    export()