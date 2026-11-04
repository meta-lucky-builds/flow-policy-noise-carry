# What Posterior Transport Policy is

This is a summary of the design. The name is historical: "posterior" means a learned conditional action distribution here, not a Bayesian posterior over the world.

## The idea

A flow-matching policy maps source noise and an observation to a chunk of H future actions, executes L of them, and replans. Standard sampling draws independent noise at each replan. PTP makes the source recurrent:

```text
Z_j = rho * zbar_j + sqrt(1 - rho^2) * eps_j,   eps_j ~ N(0, I)
```

`zbar_j` is the retained noise for action token `j`, and `rho` in [0, 0.95] is tuned on closed-loop success. Under a constant `rho` the source stays marginally N(0, I) and i.i.d. within a chunk, so a frozen decoder keeps receiving typical inputs. Conditional on the history it does not, and the specification states that plainly.

## Carry variants tested

| Mode | Anchor for token j | Tail | Memory |
|---|---|---|---|
| **shift** | Old token j+L, aligned in time | Fresh noise | Ends after H/L replans |
| **cyclic** | Old token j+L, aligned in time | Executed tokens recycled | Decays as ρ^m |
| **same-index** | Old token j, not aligned | None needed | Decays as ρ^m |

`shift` is the specification's default. `cyclic` and `same-index` came from the mechanism analysis.

## Implementation status

| Component of the specification | Status |
|---|---|
| K=1 noise carry on a frozen FM decoder | **Implemented and benchmarked** |
| Execution alignment | Implemented. The real benchmark is synchronous, using queue accounting. `point_mass/wrapper.py` has a full ledger |
| Closed-loop outcome evaluation | Implemented, with fixed settings in the completed run |
| Multi-setting search on dev tasks | Implemented, but **the only full run crashed before saving** |
| 2-D point-mass diagnostic | Implemented and tested, **never run with a trained policy** |
| Trained intents (K>1), adaptive release | Not implemented |
| Recovery imitation, latent RL, world models | Not implemented |
| Spline decoder comparator | Not implemented |
| Safety and passivity extensions | Not implemented |
| Baselines EAPN, CTP, MOSS, RTC | Not run |

The results therefore say nothing about the unimplemented components.
