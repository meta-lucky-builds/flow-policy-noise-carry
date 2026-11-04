# SmolVLA benchmark

Compares standard flow matching with noise carry on the frozen checkpoint `lerobot/smolvla_libero`. Weights, solver and LeRobot's rollout loop stay unchanged. Only the starting noise and the replanning interval differ. Versions, settings and verdict rules are fixed in [`protocol.json`](protocol.json).

## Machine

Ubuntu with root or apt access, one or more NVIDIA GPUs with 16 GB or more, 12+ CPU cores, 64 GB RAM, 150 GB disk. Setting `NVIDIA_DRIVER_CAPABILITIES=all` on the container makes GPU rendering likely to work immediately.

## Run

From the repository root:

```bash
bash ptp_vla/go.sh quick
```

```bash
python3 ptp_vla/status.py
```

`quick` runs 4 fixed settings and 80 jobs, and took 24 minutes on 4× RTX 5090. `bash ptp_vla/go.sh` with no argument runs the model check, Stage 0 and the full 14-setting search, followed by a final test on separate tasks. That full mode has **never completed**.

## Files

| File | Purpose |
|---|---|
| `carry.py` | The noise wrapper: shift, cyclic and same-index carry |
| `metrics.py` | Stage 0 instruments, paired statistics, verdict rules |
| `run.py` | Model check, Stage 0, full benchmark, quick mode |
| `parallel.py` | Workers per GPU, resource limits, crash recovery |
| `go.sh`, `setup.sh`, `fix_render.sh` | Install, repair rendering, restart |
| `status.py` | Progress and results at any time |
| `test_vla.py`, `test_parallel.py` | Tests against LeRobot's real policy class, plus scheduler failure tests |

Results are written to `ptp_vla/results/`, one job at a time, and rerunning skips finished work.

## Rendering

Setup tries GPU rendering as provided, then registers an existing NVIDIA EGL library, then installs libraries extracted from the driver package matching your driver, and finally falls back to CPU rendering through OSMesa. The backend used is recorded with the results.
