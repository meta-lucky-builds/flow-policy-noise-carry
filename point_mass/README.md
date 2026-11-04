# Point-mass diagnostic

A controlled 2-D obstacle task with two valid routes, for studying noise carry where behavior modes are known. **The code is tested, but no trained-policy result exists yet.** The first attempt was blocked by CPU speed, and the work moved to the real model.

| File | Purpose |
|---|---|
| `env.py` | Batched dynamics, swept collision checks, success rules |
| `expert.py` | A\* corridor plus convex QP, with every trajectory forward-simulated and verified |
| `gen_data.py` | 10,000 training episodes with goal, obstacle, velocity and nuisance events |
| `model.py`, `train.py` | Reference FM transformer and training |
| `wrapper.py` | Noise carry and a full execution ledger for faults and alignment |
| `stage0.py`, `bench.py`, `run_all.py` | Screen, benchmark and locked-test pipeline |
| `bench_protocol.json`, `stage0_criteria.json` | Win rules declared in advance |

## Run

Training needs a GPU. Expert data generation runs on the CPU.

```bash
pip install -r requirements.txt
```

```bash
python gen_data.py --out data.npz
```

```bash
python run_all.py --device cuda --outdir results
```

Known data limit: in the generated pilot, the expert almost never demonstrates route switching, so recovery conditions should be read with care.
