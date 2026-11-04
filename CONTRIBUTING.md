# Contributing

Contributions are welcome, and results that contradict the current findings are especially useful.

## Good contributions

- Reproductions on other checkpoints, tasks or seeds, ideally using fresh tasks
- Confirmation or refutation of the replanning and delay interaction
- Baselines that have not been run, such as EAPN, RTC or temporal ensembling
- A first trained-policy run of the `point_mass/` diagnostic
- Bug reports against the statistics or the wrapper

## Standards for results

1. Pin the code, checkpoint revision and environment.
2. Write down the settings and decision rule before running.
3. Keep tuning data and final comparison data separate.
4. Report null and failed results.
5. Include raw per-episode data, so others can recompute your numbers.

## Before opening a pull request

```bash
pip install -r requirements.txt
pytest tests
```

If you change a result or a headline number, regenerate the summary with `python analysis/analyze_results.py` and update the README in the same pull request. The tests fail if they disagree.
