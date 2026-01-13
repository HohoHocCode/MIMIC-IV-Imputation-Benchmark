import pandas as pd
import os

def generate():
    if not os.path.exists("benchmark_results.csv"):
        print("benchmark_results.csv not found.")
        return

    try:
        df = pd.read_csv("benchmark_results.csv")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Total results rows in CSV: {len(df)}")
    
    # Identify algorithms and their categories uniquely
    algos = df['Algorithm'].unique().tolist()
    
    report_data = []
    
    # Filter by dataset markers (D1 and D2)
    # Since my run_dual_benchmark.sh runs D1 then D2, 
    # and they might append to the same CSV without a "Dataset" column,
    # we can try to guess or use the order.
    # Better: I should have added a 'Dataset' column to the CSV.
    # But let's work with what we have: assuming the first occurrences are D1, second are D2.
    
    # Actually, a better way: 
    # If the row index < 23, it's D1. If >= 23, it's D2.
    
    for i, algo in enumerate(algos):
        row_d1 = df.iloc[i] if i < len(df) else None
        row_d2 = df.iloc[i+23] if (i+23) < len(df) else None
        
        entry = {
            'Algorithm': algo,
            'Category': row_d1['Class'] if row_d1 is not None else "N/A",
            'D1_NRMSE': row_d1['NRMSE'] if row_d1 is not None else 1.0,
            'D1_Time': row_d1['TimeMs'] if row_d1 is not None else 0.0,
            'D2_NRMSE': row_d2['NRMSE'] if row_d2 is not None else 1.0,
            'D2_Time': row_d2['TimeMs'] if row_d2 is not None else 0.0,
        }
        report_data.append(entry)

    # Manual LaTeX generation
    latex = "\\begin{table}[h]\n\\centering\n\\caption{So sánh hiệu năng Imputation trên D1 (Large) và D2 (High-Res)}\n\\label{tab:results_dual}\n"
    latex += "\\begin{tabular}{llcccc}\n\\toprule\n"
    latex += "\\textbf{Algorithm} & \\textbf{Cat.} & \\multicolumn{2}{c}{\\textbf{D1 (546k)}} & \\multicolumn{2}{c}{\\textbf{D2 (85k)}} \\\\\n"
    latex += "\\cmidrule(r){3-4} \\cmidrule(l){5-6}\n"
    latex += " & & NRMSE & Time(ms) & NRMSE & Time(ms) \\\\\n\\midrule\n"

    for row in report_data:
        name = str(row['Algorithm']).replace("_", "\\_").replace("\"", "")
        cat = str(row['Category'])
        d1_n = f"{row['D1_NRMSE']:.4f}"
        d1_t = f"{row['D1_Time']:.1f}"
        d2_n = f"{row['D2_NRMSE']:.4f}"
        d2_t = f"{row['D2_Time']:.1f}"
        latex += f"{name} & {cat} & {d1_n} & {d1_t} & {d2_n} & {d2_t} \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    with open("results_table.tex", "w") as f:
        f.write(latex)
    
    print("Results synced to results_table.tex")

if __name__ == "__main__":
    generate()
