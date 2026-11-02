"""Number of tokens of each label word per LLM, read from the run configurations.

A class score sums the log-probabilities of the tokens of its label, so labels of different lengths could carry a
length bias. The table records the token count of every label for every model and task.

Table: results/tables/grid_max_length_512/llm_label_token_counts.csv

    python -m src.llm_label_token_audit
"""

import json

import pandas as pd

from src.llm_evaluation import GRID_DIR, TABLES


def main():
    rows = {}
    for path in sorted((GRID_DIR / "llm_runs").glob("*/config.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        for label, tokens in zip(cfg["labels"], cfg["label_tokens"]):
            rows[(cfg["task"], cfg["model_key"], label)] = len(tokens)
    table = pd.DataFrame([{"task": t, "model": m, "label": l, "n_tokens": n} for (t, m, l), n in sorted(rows.items())])
    table.to_csv(TABLES / "llm_label_token_counts.csv", index=False, lineterminator="\n")
    print(table.pivot_table(index=["task", "model"], columns="label", values="n_tokens").to_string())


if __name__ == "__main__":
    main()
