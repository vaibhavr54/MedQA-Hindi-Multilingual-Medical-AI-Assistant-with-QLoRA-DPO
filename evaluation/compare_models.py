#!/usr/bin/env python3
"""
MedQA-Hindi Model Comparison Report Generator
Reads evaluation_results.json and produces comparison_report.md + comparison_report.html.

Fixes applied:
  1. compute_improvements() uses model_key lookup instead of positional index —
     safe when evaluate.py skipped a model (e.g. DPO not trained yet)
  2. Division-by-zero guard in improvement calculation
  3. Markdown f-string newline bug fixed (raw \n instead of escaped newline in f-strings)
  4. HTML report: sample predictions HTML-escaped to prevent broken layout
     if model output contains < > & characters
"""

import json
import html as html_lib
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
EVAL_DIR    = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
REPORT_DIR  = EVAL_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load ───────────────────────────────────────────────────────────────────────
def load_results() -> list:
    results_file = RESULTS_DIR / "evaluation_results.json"
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Run: python evaluation/evaluate.py")
        return []

    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Improvements ───────────────────────────────────────────────────────────────
def compute_improvements(results: list) -> list:
    """
    Compute % improvement of qlora and dpo over base.

    FIX: original used results[0] as base unconditionally.
    Now looks up by model_key — safe if evaluate.py skipped a model.
    Also guards against division by zero when base score is 0.
    """
    by_key = {r.get("model_key", r.get("model", "")): r for r in results}
    base   = by_key.get("base")

    if base is None:
        print("⚠️  No base model result found — skipping improvement calculation")
        return results

    def pct(fine_val, base_val):
        if base_val == 0:
            return 0.0
        return round((fine_val - base_val) / base_val * 100, 1)

    for r in results:
        if r.get("model_key", r.get("model", "")) == "base":
            continue
        r["rouge_l_improvement"]          = pct(r["rouge_l"],          base["rouge_l"])
        r["bertscore_improvement"]        = pct(r["bertscore"],        base["bertscore"])
        r["medical_accuracy_improvement"] = pct(r["medical_accuracy"], base["medical_accuracy"])

    return results


# ── Markdown report ────────────────────────────────────────────────────────────
def generate_markdown_report(results: list) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines.append("# MedQA-Hindi Model Comparison Report\n")
    lines.append(f"**Generated:** {timestamp}\n")

    # Metrics table
    lines.append("## Evaluation Metrics\n")
    lines.append("| Model | ROUGE-L | BERTScore | Medical Accuracy |")
    lines.append("|-------|---------|-----------|------------------|")
    for r in results:
        lines.append(
            f"| {r['model']} | {r['rouge_l']:.4f} | {r['bertscore']:.4f} | {r['medical_accuracy']:.1%} |"
        )

    # Improvement table (only for non-base models that have the field)
    fine_tuned = [r for r in results if "rouge_l_improvement" in r]
    if fine_tuned:
        lines.append("\n## Improvements Over Base Model\n")
        lines.append("| Model | ROUGE-L ↑ | BERTScore ↑ | Medical Accuracy ↑ |")
        lines.append("|-------|-----------|-------------|-------------------|")
        for r in fine_tuned:
            lines.append(
                f"| {r['model']} "
                f"| +{r['rouge_l_improvement']}% "
                f"| +{r['bertscore_improvement']}% "
                f"| +{r['medical_accuracy_improvement']}% |"
            )

    # Key findings
    lines.append("\n## Key Findings\n")
    if fine_tuned:
        best = max(fine_tuned, key=lambda x: x["medical_accuracy"])
        lines.append(f"- **Best performing model:** {best['model']}")
        lines.append(f"- **Highest ROUGE-L:** {best['rouge_l']:.4f}")
        lines.append(f"- **Highest BERTScore:** {best['bertscore']:.4f}")
        lines.append(f"- **Highest Medical Accuracy:** {best['medical_accuracy']:.1%}")
    else:
        lines.append("- Only base model evaluated so far.")

    # Sample predictions
    lines.append("\n## Sample Predictions\n")
    for r in results:
        lines.append(f"### {r['model']}\n")
        preds = r.get("predictions", [])[:3]
        refs  = r.get("references",  [])[:3]
        for i, (pred, ref) in enumerate(zip(preds, refs)):
            lines.append(f"**Q{i+1}:** {ref[:120]}...")
            lines.append(f"\n**A:** {pred[:180]}...\n")
            lines.append("---\n")

    return "\n".join(lines)


# ── HTML report ────────────────────────────────────────────────────────────────
def generate_html_report(results: list) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Metrics table rows
    table_rows = ""
    for r in results:
        table_rows += (
            f"<tr>"
            f"<td><strong>{html_lib.escape(r['model'])}</strong></td>"
            f"<td>{r['rouge_l']:.4f}</td>"
            f"<td>{r['bertscore']:.4f}</td>"
            f"<td>{r['medical_accuracy']:.1%}</td>"
            f"</tr>\n"
        )

    # Improvement table rows
    fine_tuned = [r for r in results if "rouge_l_improvement" in r]
    improvement_rows = ""
    for r in fine_tuned:
        improvement_rows += (
            f"<tr>"
            f"<td><strong>{html_lib.escape(r['model'])}</strong></td>"
            f"<td class='improvement'>+{r['rouge_l_improvement']}%</td>"
            f"<td class='improvement'>+{r['bertscore_improvement']}%</td>"
            f"<td class='improvement'>+{r['medical_accuracy_improvement']}%</td>"
            f"</tr>\n"
        )

    # Sample predictions
    samples_html = ""
    for r in results:
        samples_html += f"<h3>{html_lib.escape(r['model'])}</h3>\n"
        preds = r.get("predictions", [])[:3]
        refs  = r.get("references",  [])[:3]
        for i, (pred, ref) in enumerate(zip(preds, refs)):
            # FIX: escape model output — LLMs can produce <tags> that break HTML
            samples_html += (
                f"<div class='sample'>"
                f"<p class='question'>Q{i+1}: {html_lib.escape(ref[:120])}...</p>"
                f"<p class='answer'>A: {html_lib.escape(pred[:180])}...</p>"
                f"</div>\n"
            )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedQA-Hindi Model Comparison</title>
    <style>
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
            background: #0f172a;
            color: #e2e8f0;
        }}
        h1  {{ color: #38bdf8; border-bottom: 2px solid #38bdf8; padding-bottom: 0.5rem; }}
        h2  {{ color: #5eead4; margin-top: 2rem; }}
        h3  {{ color: #94a3b8; margin-top: 1.5rem; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th  {{ background: #1e293b; color: #38bdf8; padding: 12px; text-align: left; }}
        td  {{ padding: 12px; border-bottom: 1px solid #334155; }}
        tr:hover {{ background: #1e293b; }}
        .improvement {{ color: #4ade80; font-weight: bold; }}
        .timestamp   {{ color: #94a3b8; font-size: 0.9rem; }}
        .sample      {{ background: #1e293b; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
        .question    {{ color: #fbbf24; font-weight: 600; margin: 0 0 0.5rem; }}
        .answer      {{ color: #e2e8f0; margin: 0; white-space: pre-wrap; word-break: break-word; }}
    </style>
</head>
<body>
    <h1>🏥 MedQA-Hindi Model Comparison</h1>
    <p class="timestamp">Generated: {timestamp}</p>

    <h2>📊 Evaluation Metrics</h2>
    <table>
        <thead>
            <tr><th>Model</th><th>ROUGE-L</th><th>BERTScore</th><th>Medical Accuracy</th></tr>
        </thead>
        <tbody>{table_rows}</tbody>
    </table>

    <h2>📈 Improvements Over Base Model</h2>
    <table>
        <thead>
            <tr><th>Model</th><th>ROUGE-L ↑</th><th>BERTScore ↑</th><th>Medical Accuracy ↑</th></tr>
        </thead>
        <tbody>{improvement_rows if improvement_rows else "<tr><td colspan='4'>Run evaluate.py with all models to see improvements.</td></tr>"}</tbody>
    </table>

    <h2>📝 Sample Predictions</h2>
    {samples_html}
</body>
</html>
"""


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🏥 MedQA-Hindi Model Comparison Report")
    print("=" * 60)

    results = load_results()
    if not results:
        return

    results = compute_improvements(results)

    print("\n📝 Generating Markdown report...")
    md_path = REPORT_DIR / "comparison_report.md"
    md_path.write_text(generate_markdown_report(results), encoding="utf-8")
    print(f"   ✅ {md_path}")

    print("\n🌐 Generating HTML report...")
    html_path = REPORT_DIR / "comparison_report.html"
    html_path.write_text(generate_html_report(results), encoding="utf-8")
    print(f"   ✅ {html_path}")

    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"{'Model':<30} {'ROUGE-L':>10} {'BERTScore':>10} {'MedAcc':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['model']:<30} "
            f"{r['rouge_l']:>10.4f} "
            f"{r['bertscore']:>10.4f} "
            f"{r['medical_accuracy']:>10.4f}"
        )

    print("\n" + "=" * 60)
    print("✅ Reports saved to evaluation/reports/")
    print("=" * 60)


if __name__ == "__main__":
    main()