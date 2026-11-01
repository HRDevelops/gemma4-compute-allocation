# Pre-registered selector rules (written 2026-11-01, before holdout runs exist)

These rules were chosen on short seeds 0–4 and long seeds 0–2. They are frozen here
and will be evaluated unchanged on the holdout runs (short seeds 5–9, long seeds 3–4).

Valid run = patch changes at least one non-test `.py` source file in a package directory,
and the run did not end on a budget limit (time, tool calls, turns).

- **Short-run rule (R-short):** valid → last `run_command` after the final edit exited 0 → smallest patch.
  Ties broken uniformly at random; we report the exact expectation.
- **Long-run rule (R-long):** valid → last `run_command` after the final edit exited 0 →
  highest mean Jaccard similarity of normalized changed source lines to the other valid candidates.
- **Primary holdout metric:** expected tasks solved by R-short over all k-subsets (k = 2..5)
  of short seeds 5–9, vs the mean single run on the same seeds; and R-long at k = 2 over long seeds {3, 4} plus
  k = 2, 3 subsets drawn from all 5 long seeds that include at least one holdout seed.
- Code: `code/selector/select_eval.py` and `compare.py` at the time of writing.
