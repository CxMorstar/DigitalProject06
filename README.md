# DigitalProject06 - 电信客户流失预测与客户细分

## 项目简介
本项目面向课程选题 6，使用 Telco Customer Churn 数据集完成两类任务：
- 监督学习：客户流失预测（Churn Prediction）
- 无监督学习：客户分群（Customer Segmentation）

项目已提供完整 `ipynb` 文件、可复现的 Python 脚本、运行产出文件与模型文件。

## 1. 完整源代码（`.ipynb`）
- 主提交文件：`notebooks/churn_analysis.ipynb`
- 工程化脚本：`scripts/churn_analysis.py`

Notebook 和脚本实现了完整流程：数据读取、预处理、特征工程、分类建模、聚类分析、结果导出与可视化。

## 2. 依赖环境文件（`requirements.txt`）
项目依赖与版本见：`requirements.txt`。

### 安装方式
```bash
pip install -r requirements.txt
```

### 当前依赖（含版本）
- pandas==2.0.3
- numpy==1.24.3
- matplotlib==3.7.1
- seaborn==0.12.2
- scikit-learn==1.3.0
- xgboost==1.7.6
- lightgbm==3.3.2
- imbalanced-learn==0.10.1
- shap==0.42.1
- joblib==1.2.0

> 说明：若部分可选依赖无法安装，脚本支持降级运行（例如无 SMOTE 时自动使用随机过采样）。

## 3. 数据集说明（来源、规模、预处理）
### 数据来源
- Telco Customer Churn（IBM / Kaggle 同类版本）
- 本项目数据文件：`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 数据规模
- 样本数：7043
- 原始特征数：21
- 目标变量：`Churn`（Yes/No）

### 主要预处理方式
- `TotalCharges` 数值化并处理异常空白值
- 缺失值填补（数值列中位数、类别列众数）
- 关键数值列做 IQR 截尾（`tenure`、`MonthlyCharges`、`TotalCharges`）
- 类别特征 One-Hot 编码
- 构造业务特征：`service_count`、RFM 代理特征、`contract_woe`
- 类别不平衡处理：优先 SMOTE，缺失依赖时随机过采样兜底

## 4. 运行说明（从数据准备到模型推理）
### 4.1 数据准备
将原始 CSV 放入：
```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

若本地没有数据，可用（PowerShell）：
```powershell
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" `
  -OutFile "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

### 4.2 一键运行完整流程
```bash
python scripts/churn_analysis.py
```

### 4.3 Notebook 运行（课程提交推荐）
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

### 4.4 结果输出位置
- 模型文件：`models/best_model.joblib`
- 指标文件：`output/model_metrics.csv`
- 报告文件：`output/classification_reports.txt`
- 分类图表：`output/roc_curves.png`、`output/pr_curves.png`
- 分群结果：`output/customer_segments.csv`、`output/kmeans_cluster_summary.csv`
- 数据画像：`output/data_profile.json`

## 项目目录（核心）
```text
DigitalProject06/
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_analysis.ipynb
├── scripts/
│   └── churn_analysis.py
├── models/
│   └── best_model.joblib
├── output/
│   ├── model_metrics.csv
│   ├── roc_curves.png
│   ├── pr_curves.png
│   └── customer_segments.csv
├── requirements.txt
└── README.md
```
