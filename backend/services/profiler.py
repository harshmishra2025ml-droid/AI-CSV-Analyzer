import numpy as np
import pandas as pd

def _safe(v):
    if pd.isna(v) or np.isinf(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v

def profile_dataframe(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    missing = df.isna().sum()
    duplicates = int(df.duplicated().sum())
    columns = []

    for col in df.columns:
        s = df[col]
        item = {
            "name": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
            "unique_pct": round(float(s.nunique(dropna=True) / max(rows, 1) * 100), 2),
        }
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce").dropna()
            item["type"] = "numeric"
            if len(x):
                item["stats"] = {
                    "mean": _safe(x.mean()), "median": _safe(x.median()),
                    "std": _safe(x.std()), "min": _safe(x.min()),
                    "max": _safe(x.max()),
                    "q1": _safe(x.quantile(.25)), "q3": _safe(x.quantile(.75)),
                    "skew": _safe(x.skew()),
                }
                q1, q3 = x.quantile(.25), x.quantile(.75)
                iqr = q3 - q1
                item["outliers_iqr"] = int(((x < q1 - 1.5*iqr) | (x > q3 + 1.5*iqr)).sum())
        elif pd.api.types.is_datetime64_any_dtype(s):
            item["type"] = "datetime"
        else:
            item["type"] = "categorical"
            top = s.astype("string").value_counts(dropna=True).head(10)
            item["top_values"] = [{"value": str(k), "count": int(v)} for k, v in top.items()]
        columns.append(item)

    numeric = df.select_dtypes(include=np.number)
    correlations = []
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i+1:]:
                v = corr.loc[a, b]
                if pd.notna(v):
                    correlations.append({"a": str(a), "b": str(b), "correlation": round(float(v), 4)})
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    score = 100
    score -= min(35, sum(1 for c in columns if c["missing_pct"] > 20) * 7)
    score -= min(20, round((duplicates / max(rows, 1)) * 100))
    score -= min(20, len([c for c in columns if c.get("outliers_iqr", 0) > max(10, rows*.01)]) * 3)
    score = max(0, int(score))

    return {
        "rows": int(rows), "columns_count": int(cols),
        "duplicates": duplicates, "quality_score": score,
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 2),
        "columns": columns, "correlations": correlations[:30],
        "missing_total": int(missing.sum())
    }
