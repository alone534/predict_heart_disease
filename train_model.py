import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据
df = pd.read_csv('data/diabetes_data.csv')

# 2. 数据清洗
# 检查缺失值
print("缺失值情况:")
print(df.isnull().sum())
# 本数据集无缺失值，若存在缺失值可使用以下方式填充
# 例如对于数值型特征，使用均值填充
# numerical_cols = df.select_dtypes(include=[np.number]).columns
# for col in numerical_cols:
#     mean_value = df[col].mean()
#     df[col] = df[col].fillna(mean_value)
# 对于非数值型特征，使用众数填充
# non_numerical_cols = df.select_dtypes(exclude=[np.number]).columns
# for col in non_numerical_cols:
#     mode_value = df[col].mode()[0]
#     df[col] = df[col].fillna(mode_value)

# 检查异常值，简单使用箱线图查看数值型特征的异常值（这里仅打印示意，不做实际处理）
numerical_features = df.select_dtypes(include=[np.number]).columns
for col in numerical_features:
    plt.boxplot(df[col])
    plt.title(col)
    plt.savefig(f'png/{col}_boxplot.png')  # 保存图片而不是显示
    plt.clf()  # 清除当前图形

# 3. 特征工程
# 这里暂未发现需要进行编码或特征转换的特征，若存在类别型特征可进行编码
# 例如使用One - Hot编码
# categorical_cols = df.select_dtypes(include=['object']).columns
# df = pd.get_dummies(df, columns=categorical_cols)

# 4. 先使用全量特征训练模型以获取特征重要性
X_all = df.drop(columns=["PatientID", "DoctorInCharge", "Diagnosis"])
y = df["Diagnosis"]
X_train_all, X_val_all, y_train, y_val = train_test_split(
    X_all, y, test_size=0.2, random_state=42)

model_all = RandomForestClassifier(n_estimators=100, random_state=42)
model_all.fit(X_train_all, y_train)

# 获取特征重要性
feature_importances = pd.Series(model_all.feature_importances_, index=X_all.columns)

# 选择最重要的几个特征（这里选择前10个）
top_features = feature_importances.nlargest(10).index
print("选择的最重要特征:", top_features)

# 基于选择的特征划分训练/验证集
X = df[top_features]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 验证模型
y_pred = model.predict(X_val)
print("模型评估报告：")
print(classification_report(y_val, y_pred))

# 7. 保存模型
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/heart_disease_model.pkl")
print("模型已保存至 model/heart_disease_model.pkl")