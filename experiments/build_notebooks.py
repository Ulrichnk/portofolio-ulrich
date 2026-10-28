import contextlib, io, json, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
common = """from pathlib import Path
import sys
root = Path.cwd()
while not (root / "pyproject.toml").exists() and root != root.parent:
    root = root.parent
sys.path.insert(0, str(root / "src"))
"""
contents = {
    "quant-derivatives-lab": [
        (
            "Reference pricing",
            """from quantlab.black_scholes import price, greeks, implied_vol
value = price(100, 100, 1, .2, .05)
print("Call:", value)
print("Greeks:", greeks(100, 100, 1, .2, .05))
print("Recovered IV:", implied_vol(value, 100, 100, 1, .05))""",
        ),
        (
            "Independent numerical engines",
            """from quantlab.numerical import crr, monte_carlo, finite_difference
print("CRR:", crr(100, 100, 1, .2, .05, steps=800))
print("PDE:", finite_difference(100, 100, 1, .2, .05, space_steps=800))
print("Monte Carlo:", monte_carlo(100, 100, 1, .2, .05, samples=10000, steps=1, control=True))""",
        ),
        (
            "Heston and independent exotic references",
            """from quantlab.heston import Heston, heston_price
from quantlab.exotics import geometric_asian, continuous_up_out_call
print("Synthetic Heston call:", heston_price(100, 100, 1, Heston(), .03))
print("Geometric Asian:", geometric_asian(100, 100, 1, .2, .05))
print("Continuous up-and-out:", continuous_up_out_call(100, 100, 1, .2, 130, .05))""",
        ),
    ],
    "interest-rate-analytics": [
        (
            "Build and reprice the curve",
            """from quantlab.rates import bootstrap, par_swap_rate, swap_pv, bond_pv, risk_measures
curve = bootstrap([(.5, .025), (1., .027)], [([.5, 1., 1.5], .03)])
print("Discounts:", curve.discounts)
print("Swap rate:", par_swap_rate(curve, [.5, 1., 1.5]))
print("Par swap PV:", swap_pv(curve, [.5, 1., 1.5], .03))""",
        ),
        (
            "Value a bond and its rate risk",
            """value = lambda c: bond_pv(c, [.5, 1., 1.5], .035)
print("Bond and risk:", risk_measures(value, curve))
print("P&L for +100bp:", value(curve.shifted(.01))-value(curve))""",
        ),
    ],
    "financial-risk-ml": [
        (
            "Chronological model comparison",
            """from quantlab.risk import run_research
results, data, predictions, test_start = run_research(n=4000, seed=42)
print("Train/validation/test:", results["train_rows"], results["validation_rows"], results["test_rows"])
print("Selected parameters:", results["selected"])
for name, metrics in results["metrics"].items():
    print(name, "QLIKE=", round(metrics["qlike"], 6), "VaR exceptions=", metrics["var_1pct"]["exceptions"])""",
        ),
        (
            "Inspect forecast alignment",
            """print("First test origin index:", data.origins[test_start])
print("Next realized return:", data.next_returns[test_start])
print("Next squared return target:", data.y[test_start])
print("Ridge next-day variance:", predictions["ridge"][0])
print("All features at origin t use returns through t only.")""",
        ),
    ],
}
for project, items in contents.items():
    cells = []
    count = 0
    namespace = {}

    def md(s):
        cells.append(
            dict(
                cell_type="markdown",
                id=f"cell{len(cells):04}",
                metadata={},
                source=s.splitlines(True),
            )
        )

    def code(s):
        global count
        count += 1
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(s, "notebook", "exec"), namespace)
        out = (
            []
            if not buf.getvalue()
            else [
                dict(
                    output_type="stream",
                    name="stdout",
                    text=buf.getvalue().splitlines(True),
                )
            ]
        )
        cells.append(
            dict(
                cell_type="code",
                id=f"cell{len(cells):04}",
                metadata={},
                source=s.splitlines(True),
                execution_count=count,
                outputs=out,
            )
        )

    md(
        f"# {project}\n\nExecuted educational examples. Synthetic inputs. See the report for assumptions and the repository CONTRIBUTIONS.md for development provenance.\n"
    )
    code(common)
    for title, source in items:
        md("## " + title + "\n")
        code(source + "\n")
    md(
        "## Full evidence\n\nThe full protocol is `experiments/reproduce.py` at repository root. The raw tables are in `../results/` and the French research report is in `../docs/research_report.pdf`. This notebook calls the library rather than duplicating pricing algorithms.\n"
    )
    notebook = dict(
        cells=cells,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", version=sys.version.split()[0]),
        ),
        nbformat=4,
        nbformat_minor=5,
    )
    p = root / project / "notebooks"
    p.mkdir(exist_ok=True)
    (p / "research.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
    print(project, "executed", count, "cells")
