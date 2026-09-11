import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def ml_analyze(df):
    numeric = df.select_dtypes(include=np.number).dropna(axis=1, how="all")
    result = {"anomaly_detection": None, "clustering": None, "feature_notes": []}
    if numeric.shape[1] == 0:
        return result

    # Keep computation bounded for interactive use.
    sample = numeric.dropna().sample(min(len(numeric.dropna()), 10000), random_state=42) if len(numeric.dropna()) else numeric.dropna()
    if len(sample) >= 20 and sample.shape[1] >= 1:
        X = StandardScaler().fit_transform(sample)
        iso = IsolationForest(contamination="auto", random_state=42, n_estimators=150)
        labels = iso.fit_predict(X)
        result["anomaly_detection"] = {
            "sample_size": int(len(sample)),
            "anomalies": int((labels == -1).sum()),
            "anomaly_pct": round(float((labels == -1).mean()*100), 2),
        }
        if sample.shape[1] >= 2 and len(sample) >= 30:
            k = min(4, len(sample)-1)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(X)
            result["clustering"] = {
                "clusters": int(k),
                "sizes": [int(x) for x in np.bincount(cluster_labels)]
            }

    variances = numeric.var(numeric_only=True).sort_values()
    for col, val in variances.head(5).items():
        if pd.notna(val) and float(val) == 0:
            result["feature_notes"].append(f"{col} is constant and may be removable.")
    return result
