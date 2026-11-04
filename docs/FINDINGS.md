# Findings and evidence

Every SmolVLA number is recomputed from `results/smolvla_libero/quick_cells.jsonl` and `stage0_results.json` by `analysis/analyze_results.py`. Mechanism numbers are in `results/toy_mechanism.txt`.

## Experiment

| Item | Value |
|---|---|
| Model | `lerobot/smolvla_libero`, revision `a24f1f38`, frozen |
| Code | LeRobot commit `ba83e4cf`, version 0.6.2 |
| Hardware | 4× RTX 5090, 256 cores, EGL rendering, 24 simulators per worker allowed, 20 used |
| Tasks | `libero_10` and `libero_goal`, odd task indices, 10 in total |
| Conditions | Clean, and observations 4 control ticks old (0.2 s at 20 Hz) |
| Size | 4 methods × 2 conditions × 10 tasks × 20 episodes = 1,600 episodes |
| Pairing | Same seed and starting states per episode. 0 of 1,600 start-state hash mismatches |

## Success by method

| Method | Overall | Clean | Delay | `libero_10` | `libero_goal` |
|---|---|---|---|---|---|
| Native L=50 | 63.25 | 67.0 | 59.5 | 52.5 | 74.0 |
| Fresh L=10 | 67.5 | 89.5 | 45.5 | 63.5 | 71.5 |
| Shift L=10, ρ=0.9 | 68.5 | 88.5 | 48.5 | 64.5 | 72.5 |
| Same-index L=10, ρ=0.9 | 64.25 | 86.5 | 42.0 | 61.5 | 67.0 |

## Paired effects

Intervals in the first column resample tasks and episodes, and speak to tasks like these. Intervals in the second keep these tasks fixed.

| Comparison | Diff (pts) | 95%, tasks + episodes | 95%, these tasks | Wins | Exact p |
|---|---|---|---|---|---|
| Fresh L=10 − native L=50, overall | +4.25 | −5.2, +13.8 | −1.2, +9.5 | 75 / 58 | 0.165 |
| Fresh L=10 − native L=50, clean | +22.5 | +13.5, +32.0 | +16.5, +29.0 | 50 / 5 | 2.14e-10 |
| Fresh L=10 − native L=50, delay | −14.0 | −29.0, +0.5 | −22.0, −6.0 | 25 / 53 | 0.00203 |
| Shift − fresh, overall | +1.0 | −3.2, +5.5 | −2.2, +4.2 | 25 / 21 | 0.659 |
| Shift − fresh, clean | −1.0 | −5.5, +3.5 | −4.0, +2.0 | 4 / 6 | 0.754 |
| Shift − fresh, delay | +3.0 | −5.0, +11.5 | −3.0, +8.5 | 21 / 15 | 0.405 |
| Same-index − fresh, overall | −3.25 | −8.3, +2.3 | −6.8, +0.2 | 22 / 35 | 0.111 |
| Same-index − fresh, clean | −3.0 | −10.0, +3.5 | −7.0, +1.0 | 7 / 13 | 0.263 |
| Same-index − fresh, delay | −3.5 | −12.5, +5.5 | −9.0, +2.5 | 15 / 22 | 0.324 |
| Shift − same-index, overall | +4.25 | −0.2, +8.8 | +0.7, +7.8 | 36 / 19 | 0.03 |

**Multiple comparisons.** No correction was applied. The clean replanning effect easily survives any reasonable correction. The delay effect survives a Bonferroni correction across these 10 rows (0.00203 × 10 ≈ 0.02), but its interval that also resamples tasks just crosses zero. Shift versus same-index does not survive correction, so it is only suggestive. The condition split was analyzed after seeing results.

## Plan consistency

Root mean square in normalized action units across jobs. Lower means more consistent plans.

| Method | Condition | Overlap | Seam | Executed step change |
|---|---|---|---|---|
| Native L=50 | clean | n/a | 0.708 | 0.163 |
| Native L=50 | delay | n/a | 0.721 | 0.164 |
| Fresh L=10 | clean | 0.328 | 0.317 | 0.171 |
| Fresh L=10 | delay | 0.488 | 0.458 | 0.210 |
| Shift, ρ=0.9 | clean | 0.332 | 0.336 | 0.174 |
| Shift, ρ=0.9 | delay | 0.485 | 0.486 | 0.217 |
| Same-index, ρ=0.9 | clean | 0.338 | 0.327 | 0.172 |
| Same-index, ρ=0.9 | delay | 0.508 | 0.491 | 0.220 |

Carry produced no meaningful improvement over fresh noise on any of these metrics. Native L=50 has no overlap between chunks, so its overlap is undefined, and its large seam reflects 50 steps of drift between plans.

## Stage 0

Run on `libero_10` even task indices, clean only, 50 episodes per method. These tasks are separate from the benchmark's test tasks.

| Measure | Value |
|---|---|
| Decision | NO-GO |
| Success: native L=50 / fresh L=10 / shift ρ=0.85 | 46% / 56% / 54% |
| Noise spread of first 10 tokens, p10 / p50 / p90 | 0.027 / 0.038 / 0.059 |
| Token-response mass within ±5 tokens | 0.206, uniform would be 0.208 |
| Share of states flagged multimodal | 54% of 100 |
| Mode kept, shift, ρ=0 → 0.95 | 59.8% → 60.7% |
| Mode kept, cyclic, ρ=0 → 0.95 | 60.7% → 61.9% |
| Mode kept, same-index, ρ=0 → 0.95 | 62.0% → 91.0% |
| Overlap RMS, fresh → shift ρ=0.85 | 0.399 → 0.416 |

**Cautions.** Mode keeping is measured at the same observation with no state advance. Noise spread is small even where a state is flagged multimodal, so those "modes" may be small in physical terms. About 90% of the token-response mass falls on two action tokens, so the locality score is weak evidence.

## Mechanism analysis

Exact mixture-of-Gaussians flow matching, with no learned network, so these describe the mechanism and not a robot. See `results/toy_mechanism.txt`.

- **Gaussian targets.** The flow is A = m + Σ^½z (rmse 0.0013). A smooth stationary Σ makes the map local, with 98% of squared mass within ±3 tokens, so time alignment has a principled basis for such targets. SmolVLA did not show this locality.
- **Memory limit.** With H=16 and L=4, shift-carry mode agreement falls 0.746, 0.628, 0.547, 0.495 over four replans, then stays at chance. Same-index stays at 0.736 after eight.
- **Retention at ρ=0.95.** Uniform separation: shift 0.762, cyclic 0.891, same-index 0.902. Ramp separation: 0.720, 0.741, 0.897.
- **Jitter at ρ=0.95.** Shift 0.130, same-index 0.324, from about 0.42 at ρ=0.
- **State feedback.** With strong feedback (β=8), fresh noise already reverses only 0.41 times per rollout, so carry has little left to fix.
- **Selection noise.** Best of 12 null settings at n=20 inflates by +18, +13 and +5 points at true rates of 50%, 80% and 95%.

## Specification review

Math re-derived and found correct: maximal coupling and the TV bound, the Gibbs variational result, the conditional Gaussian KL, the NLL diagnostic f and f′, the TV decomposition, the contraction bound, the linear Gaussian tilt, the rational-quadratic spline derivative and the tanh Jacobian.

Structural issues found in the specification:

- The default K=1 method is itself the "simple carry" baseline its Stage B gate requires it to beat.
- Its tuning budget of 20 to 40 episodes per setting cannot resolve realistic effects.
- The tuning condition (clean or perturbed) is not stated.
- Its theorems are mostly unrelated to K=1 performance. None shows that persistence helps.
- Replanning interval and temporal ensembling are competing persistence knobs missing from its baselines.
- Latent-noise RL over a frozen policy already exists as DSRL.
