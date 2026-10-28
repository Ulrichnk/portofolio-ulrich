from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
for name in ["quant-derivatives-lab", "interest-rate-analytics", "financial-risk-ml"]:
    source = root / name / "docs/research_report.md"
    target = root / name / "docs/research_report.pdf"
    cmd = [
        "pandoc",
        str(source),
        "-o",
        str(target),
        "--pdf-engine=xelatex",
        "--resource-path=" + str(source.parent),
        "-V",
        "papersize=a4",
        "-V",
        "documentclass=article",
        "-V",
        "geometry:margin=2.2cm",
        "-V",
        "fontsize=11pt",
        "-V",
        "mainfont=DejaVu Serif",
        "-V",
        "sansfont=DejaVu Sans",
        "-V",
        "monofont=DejaVu Sans Mono",
        "-V",
        "colorlinks=true",
        "-V",
        "linkcolor=blue",
        "--number-sections",
        "-H",
        str(root / "experiments/pdf_header.tex"),
    ]
    subprocess.run(cmd, check=True)
    print(name, flush=True)
