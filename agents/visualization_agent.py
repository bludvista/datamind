import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def run_visualization_agent(eda_results: dict, output_dir: str = "outputs") -> list:
    print("🤖 Agent 3 [Visualization] — Starting...")
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []
    df = eda_results["dataframe"]

    # Chart 1 — Attrition by Department
    dept_data = eda_results["attrition_by_department"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(dept_data.keys(), dept_data.values(),
                  color=['#2196F3', '#4CAF50', '#F44336'])
    ax.set_title('Attrition Rate by Department (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Attrition Rate (%)')
    ax.set_xlabel('Department')
    for bar, val in zip(bars, dept_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    path1 = os.path.join(output_dir, "chart1_attrition_by_dept.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    chart_paths.append(path1)
    print(f"   ✅ Chart 1 saved: {path1}")

    # Chart 2 — Overtime vs Attrition
    overtime_rates = {
       'With Overtime': eda_results["overtime_vs_attrition"].get("Yes", 0),
       'Without Overtime': eda_results["overtime_vs_attrition"].get("No", 0),
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ['#F44336', '#4CAF50']
    bars = ax.bar(overtime_rates.keys(), overtime_rates.values(), color=colors, width=0.4)
    ax.set_title('Attrition Rate: Overtime vs No Overtime (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Attrition Rate (%)')
    for bar, val in zip(bars, overtime_rates.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path2 = os.path.join(output_dir, "chart2_overtime_attrition.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    chart_paths.append(path2)
    print(f"   ✅ Chart 2 saved: {path2}")

    # Chart 3 — Monthly Income Distribution
    left = df[df['Attrition'] == 'Yes']['MonthlyIncome']
    stayed = df[df['Attrition'] == 'No']['MonthlyIncome']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stayed, bins=30, alpha=0.6, color='#4CAF50', label='Stayed')
    ax.hist(left, bins=30, alpha=0.6, color='#F44336', label='Left')
    ax.set_title('Monthly Income Distribution: Left vs Stayed', fontsize=13, fontweight='bold')
    ax.set_xlabel('Monthly Income ($)')
    ax.set_ylabel('Number of Employees')
    ax.legend()
    plt.tight_layout()
    path3 = os.path.join(output_dir, "chart3_income_distribution.png")
    plt.savefig(path3, dpi=150)
    plt.close()
    chart_paths.append(path3)
    print(f"   ✅ Chart 3 saved: {path3}")

    # Chart 4 — Age Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stayed, bins=20, alpha=0.6, color='#2196F3', label='Stayed')
    ax.hist(left, bins=20, alpha=0.6, color='#FF9800', label='Left')
    left_age = df[df['Attrition'] == 'Yes']['Age']
    stayed_age = df[df['Attrition'] == 'No']['Age']
    ax.cla()
    ax.hist(stayed_age, bins=20, alpha=0.6, color='#2196F3', label=f'Stayed (avg {eda_results["avg_age_stayed"]:.1f})')
    ax.hist(left_age, bins=20, alpha=0.6, color='#FF9800', label=f'Left (avg {eda_results["avg_age_left"]:.1f})')
    ax.set_title('Age Distribution: Left vs Stayed', fontsize=13, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of Employees')
    ax.legend()
    plt.tight_layout()
    path4 = os.path.join(output_dir, "chart4_age_distribution.png")
    plt.savefig(path4, dpi=150)
    plt.close()
    chart_paths.append(path4)
    print(f"   ✅ Chart 4 saved: {path4}")

    print(f"✅ Agent 3 Done! {len(chart_paths)} charts saved to '{output_dir}/'")
    return chart_paths


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from agents.ingestion_agent import run_ingestion_agent
    from agents.eda_agent import run_eda_agent

    ingestion = run_ingestion_agent("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    eda = run_eda_agent(ingestion)
    run_visualization_agent(eda)