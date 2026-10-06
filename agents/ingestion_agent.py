from pathlib import Path

import pandas as pd

def run_ingestion_agent(filepath: str) -> dict:
    print("🤖 Agent 1 [Ingestion] — Starting...")

    df = pd.read_csv(filepath)

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    missing = df.isnull().sum().sum()

    result = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": int(missing),
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "dataframe": df
    }

    print(f"✅ Agent 1 Done!")
    print(f"   Rows     : {result['total_rows']}")
    print(f"   Columns  : {result['total_columns']}")
    print(f"   Missing  : {result['missing_values']}")
    print(f"   Numerical: {len(numerical_cols)} columns")
    print(f"   Category : {categorical_cols}")
    return result


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    run_ingestion_agent(str(project_root / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"))