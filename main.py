from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 模拟读取模型
# model = joblib.load("model/heart_disease_model.pkl")

# API 初始化
app = FastAPI(
    title="心脏病预测API(取前10大影响特征)",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

class HeartDiseaseInput(BaseModel):
    FastingBloodSugar: float
    HbA1c: float
    DietQuality: float
    SerumCreatinine: float
    MedicalCheckupsFrequency: float
    BMI: float
    MedicationAdherence: float
    CholesterolHDL: float
    CholesterolTriglycerides: float
    SystolicBP: int

# 定义特征列
FEATURE_COLUMNS = ['FastingBloodSugar', 'HbA1c', 'DietQuality', 'SerumCreatinine',
                   'MedicalCheckupsFrequency', 'BMI', 'MedicationAdherence',
                   'CholesterolHDL', 'CholesterolTriglycerides', 'SystolicBP']

@app.post("/predict")
def predict_heart_disease(data: HeartDiseaseInput):
    try:
        # 将输入数据转换为特征矩阵
        X = np.array([[getattr(data, field) for field in FEATURE_COLUMNS]])
        print("🧪 输入维度:", X.shape)

        # 模拟使用模型进行预测
        # pred = model.predict(X)[0]
        # prob = model.predict_proba(X)[0][int(pred)]
        pred = 0
        prob = 0.5

        # 根据概率得出风险评分
        risk = "高" if prob > 0.7 else ("中" if prob > 0.4 else "低")
        return {
            "预测心脏病": "是" if pred == 1 else "否",
            "患病概率": round(prob, 3),
            "风险评分": risk
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }