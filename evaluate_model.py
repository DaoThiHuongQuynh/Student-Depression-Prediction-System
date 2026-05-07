import json
import math
import os
import csv
import random
import statistics
import time

# --- CẤU HÌNH ---
DATASET_FILE = "Student Depression Dataset.csv"
PARAMS_FILE = "model_params.json"


class EvaluationEngine:
    def __init__(self):
        self.params = self.load_params()

    def load_params(self):
        if os.path.exists(PARAMS_FILE):
            try:
                with open(PARAMS_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        print(f"⚠️ Cảnh báo: Không tìm thấy {PARAMS_FILE}, dùng tham số mặc định.")
        return {
            "means": [20, 3.0, 3, 3, 3, 1, 0.5, 0.5],
            "stds": [5, 0.5, 1, 1, 1, 1, 0.5, 0.5],
            "coefficients": [0.1, -0.5, 0.8, -0.4, 0.6, -0.3, 0.1, 0.5],
            "intercept": -1.5
        }

    def _convert_input(self, val, type_func, default):
        try:
            return type_func(val)
        except:
            return default

    def predict_proba(self, inputs):
        # Logic dự đoán giống hệt app chính để đảm bảo nhất quán
        age = self._convert_input(inputs.get('age'), float, 20.0)
        cgpa = self._convert_input(inputs.get('cgpa'), float, 3.0)
        ap = self._convert_input(inputs.get('academic_pressure'), float, 3.0)
        ss = self._convert_input(inputs.get('study_satisfaction'), float, 3.0)
        fs = self._convert_input(inputs.get('financial_stress'), float, 3.0)

        g_val = inputs.get('gender', 0)
        gender = 1.0 if isinstance(g_val, str) and g_val.lower() in ["nam", "male"] else (
            float(g_val) if not isinstance(g_val, str) else 0.0)

        fh_val = inputs.get('family_history', 0)
        fh = 1.0 if isinstance(fh_val, str) and fh_val.lower() in ["có", "yes", "1"] else (
            float(fh_val) if not isinstance(fh_val, str) else 0.0)

        s_val = inputs.get('sleep_duration', 1.0)
        if isinstance(s_val, str):
            s_str = s_val.lower()
            if "dưới 5" in s_str or "less than 5" in s_str:
                sd = 0.0
            elif "5-6" in s_str:
                sd = 1.0
            elif "7-8" in s_str:
                sd = 2.0
            elif "trên 8" in s_str or "more than 8" in s_str:
                sd = 3.0
            else:
                sd = 1.0
        else:
            sd = float(s_val)

        raw_features = [age, cgpa, ap, ss, fs, sd, gender, fh]
        z_scores = []
        for i, val in enumerate(raw_features):
            mean = self.params['means'][i]
            std = self.params['stds'][i]
            if std == 0: std = 1
            z_scores.append((val - mean) / std)

        logit = self.params['intercept']
        for i, z in enumerate(z_scores):
            logit += z * self.params['coefficients'][i]

        return 1.0 / (1.0 + math.exp(-logit))

    def load_data(self):
        data = []
        if not os.path.exists(DATASET_FILE):
            print(f"❌ Lỗi: Không tìm thấy file '{DATASET_FILE}'")
            return None, "None"

        print(f"📂 Đang đọc dữ liệu từ: {DATASET_FILE} ...")
        try:
            with open(DATASET_FILE, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    try:
                        item = {
                            'age': row.get('Age', 20),
                            'cgpa': row.get('CGPA', 3.0),
                            'academic_pressure': row.get('Academic Pressure', 3),
                            'study_satisfaction': row.get('Study Satisfaction', 3),
                            'financial_stress': row.get('Financial Stress', 3),
                            'gender': row.get('Gender', 'Male'),
                            'family_history': row.get('Family History of Mental Illness', 'No'),
                            'sleep_duration': row.get('Sleep Duration', '7-8 hours'),
                        }
                        dep_val = row.get('Depression', '0')
                        label = 1 if dep_val in ['1', 'Yes', 'yes', 'True'] else 0
                        data.append((item, label))
                        count += 1
                    except ValueError:
                        continue
            print(f"✅ Đã tải thành công {count} dòng dữ liệu.")
            return data, "Real Data"
        except Exception as e:
            print(f"❌ Lỗi đọc file: {e}")
            return None, "Error"

    def calculate_metrics(self, y_true, y_pred):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        accuracy = (tp + tn) / (len(y_true) or 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return accuracy, f1, precision, recall

    def run(self):
        print("\n" + "=" * 50)
        print("   HỆ THỐNG ĐÁNH GIÁ SỨC KHỎE MÔ HÌNH AI")
        print("=" * 50 + "\n")

        dataset, source = self.load_data()

        if not dataset:
            print("⚠️ Không có dữ liệu để đánh giá.")
            return

        print("\n🚀 Đang chạy Bootstrap Evaluation (50 iterations)...")
        time.sleep(1)

        stats = {'acc': [], 'f1': [], 'prec': [], 'rec': []}

        # Thanh loading đơn giản
        total = 50
        for i in range(total):
            sample = [random.choice(dataset) for _ in range(len(dataset))]
            y_true = [item[1] for item in sample]
            y_pred = []
            for item in sample:
                prob = self.predict_proba(item[0])
                y_pred.append(1 if prob >= 0.5 else 0)

            acc, f1, prec, rec = self.calculate_metrics(y_true, y_pred)
            stats['acc'].append(acc)
            stats['f1'].append(f1)
            stats['prec'].append(prec)
            stats['rec'].append(rec)

            # Print loading bar
            percent = (i + 1) * 100 // total
            bar = '█' * (percent // 5) + '-' * (20 - percent // 5)
            print(f"\rProcess: |{bar}| {percent}%", end="")

        print("\n\n" + "-" * 50)
        print("📊 BÁO CÁO KẾT QUẢ CHI TIẾT")
        print("-" * 50)

        # Tính toán kết quả
        res = {}
        for k in stats:
            res[k] = (statistics.mean(stats[k]), statistics.stdev(stats[k]))

        # Đánh giá độ tin cậy
        score = (res['acc'][0] * 0.6) + (res['f1'][0] * 0.4) - (res['acc'][1] * 2)
        score = max(0.0, min(1.0, score))
        if score > 0.85:
            reliability = "RẤT CAO (EXCELLENT)"
        elif score > 0.75:
            reliability = "CAO (GOOD)"
        elif score > 0.60:
            reliability = "TRUNG BÌNH (FAIR)"
        else:
            reliability = "THẤP (POOR)"

        # In bảng kết quả đẹp
        print(f"{'METRIC':<15} | {'MEAN':<10} | {'STD DEV (±)':<10}")
        print("-" * 43)
        print(f"{'Accuracy':<15} | {res['acc'][0]:.1%}      | {res['acc'][1]:.1%}")
        print(f"{'F1-Score':<15} | {res['f1'][0]:.1%}      | {res['f1'][1]:.1%}")
        print(f"{'Precision':<15} | {res['prec'][0]:.1%}      | {res['prec'][1]:.1%}")
        print(f"{'Recall':<15} | {res['rec'][0]:.1%}      | {res['rec'][1]:.1%}")
        print("-" * 43)
        print(f"\n🌟 ĐỘ TIN CẬY MÔ HÌNH: {score:.1%} - {reliability}")
        print("=" * 50 + "\n")

        input("Nhấn Enter để thoát...")


if __name__ == "__main__":
    engine = EvaluationEngine()
    engine.run()