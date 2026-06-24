"""End-to-end Telco customer churn mining pipeline.

Usage:
    python scripts/churn_analysis.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"

try:
    from imblearn.over_sampling import SMOTE

    HAS_SMOTE = True
except Exception:
    # 允许在离线环境运行：无 imblearn 时走随机过采样兜底。
    HAS_SMOTE = False


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # 目标变量二值化：Yes/No -> 1/0
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    # TotalCharges often contains spaces that should be treated as missing.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Binary normalization for easier downstream processing.
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        data[col] = data[col].map({"Yes": 1, "No": 0})

    data["gender"] = data["gender"].map({"Female": 0, "Male": 1})

    # 缺失值处理：数值列用中位数，类别列用众数。
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    cat_cols = data.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mode(dropna=True).iloc[0])

    # 对关键数值列做 IQR 截尾，降低异常点对模型训练的干扰。
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        data[col] = data[col].clip(lower=low, upper=high)

    # 构造 RFM 代理特征，增强业务可解释性（并非严格交易型 RFM）。
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    data["service_count"] = (
        data[service_cols]
        .astype(str)
        .apply(lambda row: np.sum(row.isin(["Yes", "DSL", "Fiber optic"])), axis=1)
    )
    data["rfm_recency_proxy"] = 72 - data["tenure"]
    data["rfm_frequency_proxy"] = data["service_count"]
    data["rfm_monetary"] = data["TotalCharges"]

    data["avg_monthly_spend"] = data["TotalCharges"] / np.maximum(data["tenure"], 1)
    data["avg_monthly_spend"] = data["avg_monthly_spend"].replace([np.inf, -np.inf], 0)

    # 使用合约类型的 WOE 编码，增强线性模型可解释性。
    data["contract_woe"] = map_woe(data["Contract"], data["Churn"])

    return data


def map_woe(feature: pd.Series, target: pd.Series, eps: float = 1e-6) -> pd.Series:
    # WOE = ln(正类占比/负类占比)，eps 用于防止分母为 0。
    frame = pd.DataFrame({"feature": feature.astype(str), "target": target})
    grp = frame.groupby("feature")["target"]
    pos = grp.sum()
    total = grp.count()
    neg = total - pos

    pos_rate = (pos + eps) / (pos.sum() + eps)
    neg_rate = (neg + eps) / (neg.sum() + eps)
    woe = np.log(pos_rate / neg_rate)
    return feature.astype(str).map(woe.to_dict()).astype(float)


def prepare_supervised_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # 去掉目标列和纯标识列（customerID），其余特征统一 One-Hot。
    y = df["Churn"].astype(int)
    drop_cols = ["Churn", "customerID"]
    X = df.drop(columns=drop_cols)
    X = pd.get_dummies(X, drop_first=False)
    return X, y


def oversample_training_data(
    X_train: np.ndarray, y_train: pd.Series
) -> tuple[np.ndarray, np.ndarray, str]:
    if HAS_SMOTE:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
        return X_bal, y_bal, "smote"

    # 兜底方案：随机过采样（当无法安装 SMOTE 依赖时仍可完成实验）。
    y_arr = np.asarray(y_train)
    classes, counts = np.unique(y_arr, return_counts=True)
    maj_class = classes[np.argmax(counts)]
    min_class = classes[np.argmin(counts)]

    maj_idx = np.where(y_arr == maj_class)[0]
    min_idx = np.where(y_arr == min_class)[0]
    add_idx = np.random.default_rng(RANDOM_STATE).choice(
        min_idx, size=len(maj_idx) - len(min_idx), replace=True
    )
    all_idx = np.concatenate([maj_idx, min_idx, add_idx])
    np.random.default_rng(RANDOM_STATE).shuffle(all_idx)
    return X_train[all_idx], y_arr[all_idx], "random_oversample"


def get_models() -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
            "scale": True,
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
            ),
            "scale": False,
        },
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = {
            "model": XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale": False,
        }
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = {
            "model": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=RANDOM_STATE,
            ),
            "scale": False,
        }
    except Exception:
        pass

    return models


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    # 分层切分，保证训练集/测试集标签比例一致。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model_specs = get_models()
    sampling_method = "none"

    metrics_records: list[dict[str, Any]] = []
    reports: dict[str, str] = {}
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    pr_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    trained_artifacts: dict[str, dict[str, Any]] = {}

    for name, spec in model_specs.items():
        model = spec["model"]
        scale = spec["scale"]

        scaler = None
        X_train_used = X_train.values
        X_test_used = X_test.values

        if scale:
            # 对需要尺度敏感的模型做标准化（如逻辑回归）。
            scaler = StandardScaler()
            X_train_used = scaler.fit_transform(X_train_used)
            X_test_used = scaler.transform(X_test_used)

        X_train_bal, y_train_bal, sampling_method = oversample_training_data(
            X_train_used, y_train
        )
        model.fit(X_train_bal, y_train_bal)

        y_pred = model.predict(X_test_used)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_used)[:, 1]
        else:
            y_score = model.decision_function(X_test_used)

        roc_auc = roc_auc_score(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score)

        metrics_records.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

        reports[name] = classification_report(y_test, y_pred, digits=4)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        roc_data[name] = (fpr, tpr, roc_auc)
        pr_data[name] = (rec, prec, pr_auc)

        trained_artifacts[name] = {
            "model": model,
            "scaler": scaler,
            "X_test_used": X_test_used,
            "y_test": y_test.values,
        }

    metrics_df = pd.DataFrame(metrics_records).sort_values("roc_auc", ascending=False)
    best_model_name = metrics_df.iloc[0]["model"]

    return {
        "metrics_df": metrics_df,
        "reports": reports,
        "roc_data": roc_data,
        "pr_data": pr_data,
        "trained_artifacts": trained_artifacts,
        "best_model_name": best_model_name,
        "X_columns": X.columns.tolist(),
        "sampling_method": sampling_method,
    }


def save_classification_outputs(results: dict[str, Any], X_columns: list[str]) -> None:
    metrics_df = results["metrics_df"]
    reports = results["reports"]
    roc_data = results["roc_data"]
    pr_data = results["pr_data"]
    artifacts = results["trained_artifacts"]
    best_name = results["best_model_name"]

    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    with (OUTPUT_DIR / "classification_reports.txt").open("w", encoding="utf-8") as f:
        for model_name, report in reports.items():
            f.write(f"===== {model_name} =====\n")
            f.write(report)
            f.write("\n\n")

    # 输出ROC/PR图，满足模型对比展示需求。
    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr, auc_val) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for model_name, (rec, prec, auc_val) in pr_data.items():
        plt.plot(rec, prec, label=f"{model_name} (AP={auc_val:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pr_curves.png", dpi=200)
    plt.close()

    best_artifact = artifacts[best_name]
    model_bundle = {
        "model_name": best_name,
        "model": best_artifact["model"],
        "scaler": best_artifact["scaler"],
        "feature_columns": X_columns,
    }
    joblib.dump(model_bundle, MODELS_DIR / "best_model.joblib")

    # 若最佳模型支持特征重要性，则额外导出特征贡献表。
    if hasattr(best_artifact["model"], "feature_importances_"):
        imp = pd.DataFrame(
            {
                "feature": X_columns,
                "importance": best_artifact["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        imp.to_csv(OUTPUT_DIR / "feature_importance_best_model.csv", index=False)

    save_shap_plot(best_name, best_artifact, X_columns)


def save_shap_plot(model_name: str, artifact: dict[str, Any], X_columns: list[str]) -> None:
    try:
        import shap
    except Exception:
        return

    model = artifact["model"]
    X_test = artifact["X_test_used"]
    X_sample = X_test[: min(500, len(X_test))]

    # SHAP 仅对树模型执行，避免在线性模型上引入不必要复杂度。
    if model_name not in {"random_forest", "xgboost", "lightgbm"}:
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=X_columns, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        return


def run_clustering(df: pd.DataFrame) -> None:
    seg_df = df.copy()

    segment_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "service_count",
        "rfm_recency_proxy",
        "rfm_frequency_proxy",
        "rfm_monetary",
        "avg_monthly_spend",
        "SeniorCitizen",
        "Partner",
        "Dependents",
    ]

    X_seg = seg_df[segment_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    # 通过轮廓系数在 2~8 类间选择较优K值。
    silhouette_records: list[dict[str, Any]] = []
    best_k = 2
    best_score = -1.0
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_records.append({"k": k, "silhouette": score})
        if score > best_score:
            best_score = score
            best_k = k

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    seg_df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

    # DBSCAN 作为密度聚类对照，用于观察离群与稀疏簇结构。
    dbscan = DBSCAN(eps=1.2, min_samples=25)
    seg_df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

    summary = (
        seg_df.groupby("kmeans_cluster")[segment_cols + ["Churn"]]
        .mean()
        .reset_index()
        .sort_values("Churn", ascending=False)
    )

    pd.DataFrame(silhouette_records).to_csv(
        OUTPUT_DIR / "kmeans_silhouette_scores.csv", index=False
    )
    summary.to_csv(OUTPUT_DIR / "kmeans_cluster_summary.csv", index=False)
    seg_df.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

    # PCA 降维后绘制 2D 散点图
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    viz = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "kmeans_cluster": seg_df["kmeans_cluster"],
            "dbscan_cluster": seg_df["dbscan_cluster"],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sample = viz.sample(min(2000, len(viz)), random_state=RANDOM_STATE)
    axes[0].scatter(
        sample["pc1"],
        sample["pc2"],
        c=sample["kmeans_cluster"],
        cmap="tab10",
        s=14,
        alpha=0.8,
    )
    axes[0].set_title("KMeans Segmentation")

    axes[1].scatter(
        sample["pc1"],
        sample["pc2"],
        c=sample["dbscan_cluster"],
        cmap="tab10",
        s=14,
        alpha=0.8,
    )
    axes[1].set_title("DBSCAN Segmentation")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "clustering_scatter.png", dpi=220)
    plt.close()


def save_data_profile(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    profile = {
        "raw_shape": list(df_raw.shape),
        "clean_shape": list(df_clean.shape),
        "target_distribution": df_clean["Churn"].value_counts().to_dict(),
        "missing_after_cleaning": int(df_clean.isna().sum().sum()),
        "columns": df_clean.columns.tolist(),
    }
    with (OUTPUT_DIR / "data_profile.json").open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def main() -> None:
    ensure_dirs()

    raw_df = load_data(DATA_PATH)
    clean_df = clean_and_engineer(raw_df)
    clean_df.to_csv(OUTPUT_DIR / "processed_telco.csv", index=False)

    save_data_profile(raw_df, clean_df)

    X, y = prepare_supervised_data(clean_df)
    results = train_and_evaluate(X, y)
    save_classification_outputs(results, results["X_columns"])

    run_clustering(clean_df)

    print("Pipeline completed.")
    print(f"Best model: {results['best_model_name']}")
    print(f"Sampling method: {results['sampling_method']}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
