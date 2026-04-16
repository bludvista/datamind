from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def run_insight_agent(eda_results: dict, rag_context: str = "") -> dict:
    print("🤖 Agent 4 [Insight] — Starting...")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    prompt = f"""
You are a senior HR data analyst. Analyze this IBM HR Attrition dataset and write a professional report.

Key Statistics:
- Overall Attrition Rate: {eda_results['attrition_rate']}%
- Attrition by Department: {eda_results['attrition_by_department']}
- Overtime vs Attrition: {eda_results['overtime_vs_attrition']}
- Average Age of employees who LEFT: {eda_results['avg_age_left']}
- Average Age of employees who STAYED: {eda_results['avg_age_stayed']}

Relevant HR Research Context:
{rag_context if rag_context else "No additional context available."}

Write 3 sections:
1. KEY FINDINGS (3 bullet points with specific numbers)
2. ROOT CAUSES (2-3 sentences explaining why these patterns exist)
3. RECOMMENDATIONS (3 actionable bullet points for HR management)

Be specific, use the numbers provided, keep it professional.
"""

    response = client.chat.completions.create(
        model="openrouter/auto",
        messages=[{"role": "user", "content": prompt}]
    )
    insights = response.choices[0].message.content

    print("✅ Agent 4 Done! LLM insights generated.")
    print("\n--- INSIGHTS PREVIEW ---")
    print(insights[:300] + "...")

    return {
        "insights": insights,
        "eda_results": eda_results
    }


if __name__ == "__main__":
    from ingestion_agent import run_ingestion_agent
    from eda_agent import run_eda_agent
    from rag_agent import run_rag_agent

    data = run_ingestion_agent("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    eda = run_eda_agent(data)
    rag = run_rag_agent(eda)
    run_insight_agent(eda, rag_context=rag["rag_context"])