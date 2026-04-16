import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.ingestion_agent import run_ingestion_agent

def run_eda_agent(ingestion_result: dict) -> dict:
    print("\n🤖 Agent 2 [EDA] — Starting...")

    df = ingestion_result["dataframe"]

    # Overall attrition rate
    attrition_rate = round((df['Attrition'].value_counts()['Yes'] / len(df)) * 100, 2)

    # Attrition by Department
    dept_attrition = df.groupby('Department')['Attrition'].apply(
        lambda x: round((x == 'Yes').sum() / len(x) * 100, 2)
    ).to_dict()

    # Average Monthly Income by Department
    avg_income = df.groupby('Department')['MonthlyIncome'].mean().round(2).to_dict()

    # Overtime vs Attrition
    overtime_attrition = df.groupby('OverTime')['Attrition'].apply(
        lambda x: round((x == 'Yes').sum() / len(x) * 100, 2)
    ).to_dict()

    # Job Satisfaction vs Attrition
    satisfaction_attrition = df.groupby('JobSatisfaction')['Attrition'].apply(
        lambda x: round((x == 'Yes').sum() / len(x) * 100, 2)
    ).to_dict()

    # Age statistics
    avg_age_left = round(df[df['Attrition'] == 'Yes']['Age'].mean(), 2)
    avg_age_stayed = round(df[df['Attrition'] == 'No']['Age'].mean(), 2)

    result = {
        "attrition_rate": attrition_rate,
        "attrition_by_department": dept_attrition,
        "avg_income_by_department": avg_income,
        "overtime_vs_attrition": overtime_attrition,
        "satisfaction_vs_attrition": satisfaction_attrition,
        "avg_age_left": avg_age_left,
        "avg_age_stayed": avg_age_stayed,
        "dataframe": df
    }

    print(f"✅ Agent 2 Done!")
    print(f"   Overall Attrition Rate : {attrition_rate}%")
    print(f"   Attrition by Dept      : {dept_attrition}")
    print(f"   Overtime (Yes) Rate    : {overtime_attrition.get('Yes', 0)}%")
    print(f"   Overtime (No) Rate     : {overtime_attrition.get('No', 0)}%")
    print(f"   Avg Age (Left)         : {avg_age_left}")
    print(f"   Avg Age (Stayed)       : {avg_age_stayed}")
    return result


if __name__ == "__main__":
    filepath = "D:\\datamind\\data\\WA_Fn-UseC_-HR-Employee-Attrition.csv"
    ingestion_result = run_ingestion_agent(filepath)
    run_eda_agent(ingestion_result)