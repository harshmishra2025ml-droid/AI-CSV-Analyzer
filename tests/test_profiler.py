import pandas as pd
from backend.services.profiler import profile_dataframe

def test_profile():
    df = pd.DataFrame({"age":[20,21,None], "name":["a","b","b"]})
    p = profile_dataframe(df)
    assert p["rows"] == 3
    assert p["columns_count"] == 2
    assert p["missing_total"] == 1
