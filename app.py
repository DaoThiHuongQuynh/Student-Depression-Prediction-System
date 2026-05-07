import os
import json
import math
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- LOAD THAM SỐ MÔ HÌNH ---
model_data = None
json_path = "model_params.json"

if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model_data:
        return jsonify({"error": "Không tìm thấy file model_params.json"}), 500

    try:
        data = request.json
        # Lấy dữ liệu từ frontend
        age = float(data['age'])
        cgpa = float(data['cgpa'])
        ap = float(data['ap'])  # Áp lực học tập
        ss = float(data['ss'])  # Hài lòng việc học
        fs = float(data['fs'])  # Áp lực tài chính
        gender = 1.0 if data['gender'] == "Male" else 0.0
        fh = 1.0 if data['fh'] == "Yes" else 0.0

        sd_map = {"Less than 5 hours": 0.0, "5-6 hours": 1.0, "7-8 hours": 2.0, "More than 8 hours": 3.0}
        sd = sd_map.get(data['sd'], 1.0)

        # Vector đặc trưng
        raw_features = [age, cgpa, ap, ss, fs, sd, gender, fh]

        means = model_data["means"]
        stds = model_data["stds"]
        coeffs = model_data["coefficients"]
        intercept = model_data["intercept"]

        logit = intercept
        for i in range(len(raw_features)):
            z = (raw_features[i] - means[i]) / stds[i] if stds[i] != 0 else 0.0
            logit += z * coeffs[i]

        probability = 1.0 / (1.0 + math.exp(-logit))
        percent = int(probability * 100)

        # Trả về kết quả và chi tiết rủi ro để vẽ biểu đồ
        return jsonify({
            "probability": probability,
            "percent": percent,
            "status": "NGUY CƠ CAO" if probability >= 0.5 else "AN TOÀN",
            "risk_details": {
                "ap": ap / 5.0,  # Áp lực học
                "fs": fs / 5.0,  # Tài chính
                "sd": (3.0 - sd) / 3.0,  # Thiếu ngủ
                "ss": (5.0 - ss) / 4.0  # Chán học
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)