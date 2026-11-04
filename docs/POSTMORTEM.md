# Post-mortem: the run that lost its results

## What happened

The first full benchmark started 8 workers with up to 20 simulators each. Seconds later, the main process was killed from outside Python ([log excerpt](../results/smolvla_libero/crashed_run_excerpt.log)):

```text
08:30:02 dev cells 220 to run 220
bash: line 1:  7485 Killed                  python run.py all
```

The workers were separate processes and kept going. The log shows they finished nearly all 220 settings jobs, but only the main process saved results, so every one of them was lost. The final test could not start, and the machine then sat idle.

The log's 213 tracebacks were `BrokenPipeError` from simulators shutting down afterward. They were a symptom.

## Root cause

**Confirmed:** the design had a single point of failure for saving results, and workers outlived their parent.

**Not confirmed:** why the process was killed. Running out of memory was the first guess, but the machine had about 500 GB of RAM, and the log holds no kernel message. The later run printed this diagnostic information but never needed it, because it was not killed.

## What changed

| Problem | Fix |
|---|---|
| Parent was the only writer | Workers save each job with a file lock and `fsync` |
| A crash lost everything | Restart skips finished jobs and tolerates a cut-off last line |
| Workers ran unattended | They stop when the parent disappears |
| Stuck jobs waited 6 hours | The limit is now 40 minutes |
| Simulator count was fixed by cores | Limited by measured RAM and GPU memory, with one worker per GPU |
| A crash ended the run | Automatic restart, up to 6 attempts, printing memory state each time |
| 20 MB log of progress bars | Progress bars and repeated warnings removed |

The next run finished 80 of 80 jobs on the first attempt with no restarts.

## What it cost scientifically

The lost jobs averaged about 49% success, but the log does not attribute them to methods, so they were not used. The full search over 14 settings has still never completed. The reported result comes from a smaller run with four fixed settings. Three same-index settings were added to the full protocol after Stage 0, and this is recorded as `_revision_1` in [`protocol.json`](../ptp_vla/protocol.json).
