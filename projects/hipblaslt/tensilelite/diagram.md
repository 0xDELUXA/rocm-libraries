# CPU Reference Pipeline — Timeline Diagram

## Baseline (Synchronous, No Pipeline)

Everything runs on the main thread, sequentially. SolveCPU blocks the entire iteration.

```
Problem 0                        Problem 1                        Problem 2
|                                |                                |
Main:  [dataInit][SolveCPU][GPU+V][dataInit][SolveCPU][GPU+V][dataInit][SolveCPU][GPU+V]
                 |<-96ms->|      |         |<-96ms->|
```

## Pipeline (2-Deep Ring Buffer, Current Implementation)

Worker thread runs SolveCPU asynchronously. Main thread posts tasks after rotating_buffer_prep,
then continues with the solution loop. The wait only happens at `preProblem` of the NEXT iteration.

```
ITERATION 0 (sync)                    ITERATION 1                                      ITERATION 2
|                                     |                                                |
Main:  [dataInit+SolveCPU(0)][gpuPrep][rotBuf][precomp 1,2][--solutions 0 (x31)--][WAIT 1][gpuPrep][rotBuf][precomp 3][--solutions 1 (x31)--][WAIT 2]...
       |<--- sync ~100ms --->|        |       |  post 1,2  | kernel+validate x31 |       |        |       | post 3   | kernel+validate x31 |
                                               |            |<-- worker overlaps -->|      |                           |<-- worker overlaps -->|
Worker:                                        [SolveCPU(1) ========================]      [SolveCPU(2) ===============][SolveCPU(3)...]
                                               |<------------ ~141ms ------------->|
```

### What SolveCPU overlaps with (per iteration)

```
                        Main thread work AFTER posting task, BEFORE next wait
                        ========================================================

startPrecompute:                               Solutions loop (per solution x 31):
  prepareCPUInputs(N+1)     ~1ms                 gpu_input_reset     ~0.0ms
  deepCopy(N+1)             ~1ms                 kernel_solving      ~0.003ms
  prepareCPUInputs(N+2)     ~1ms                 warmup launch       ~0.01ms
  deepCopy(N+2)             ~1ms                 validate_warmups:
                            ----                   gpu_readback      ~0.32ms
                            ~4ms                   element_compare   ~0.02ms
                                                 post_solution       ~0.01ms
                                                                     --------
                                                              x31 =  ~11ms

                                                post_problem          ~0.08ms

                    TOTAL OVERLAP WINDOW: ~4ms + ~11ms = ~15ms
                    ==========================================

If SolveCPU finishes within 15ms  → wait = 0        (FULLY OVERLAPPED)
If SolveCPU takes 50ms            → wait = ~35ms     (PARTIAL OVERLAP)
If SolveCPU takes 141ms           → wait = ~126ms    (BOTTLENECK)
```

### Ring buffer state across iterations

```
Iter 0 (sync):   slots: [empty] [empty]     sync SolveCPU(0) on main thread
  post 1,2:      slots: [  1  ] [  2  ]     worker starts SolveCPU(1)

  -- solutions loop for problem 0 runs here (~15ms) --
  -- worker overlaps: SolveCPU(1) running concurrently --

Iter 1:          consume slot 0             slots: [empty] [  2  ]
                 (wait if SolveCPU(1) not done yet)
  post 3:        slots: [  3  ] [  2  ]     worker finishes 1, starts SolveCPU(2)

  -- solutions loop for problem 1 runs here (~15ms) --

Iter 2:          consume slot 1             slots: [  3  ] [empty]
                 (wait if SolveCPU(2) not done yet)
  post 4:        slots: [  3  ] [  4  ]     worker finishes 2, starts SolveCPU(3)
```

### When the pipeline helps vs hurts

```
HELPS (SolveCPU ≤ overlap window ~15ms):
  Main:  [...precompute...][--solutions loop ~15ms--]
  Worker:     [SolveCPU ~12ms ~~~]  (done before next wait)
  Wait: 0ms   Saved: ~12ms per problem

PARTIAL (SolveCPU moderately larger):
  Main:  [...precompute...][--solutions loop ~15ms--][wait ~35ms]
  Worker:     [SolveCPU ~50ms ~~~~~~~~~~~~~~~~~~~~~~~~~~]
  Wait: 35ms  Saved: ~15ms per problem (still faster than sync by 15ms)

BOTTLENECK (SolveCPU >> overlap window):
  Main:  [...precompute...][--solutions loop ~15ms--][wait ~~~126ms~~~]
  Worker:     [SolveCPU ~141ms ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]
  Wait: 126ms Saved: ~15ms per problem (only 9% improvement + deep copy overhead)
```

### Key insight

The overlap window is ~15ms per iteration (GPU prep + solutions loop with validate_warmups).
The pipeline saves at most ~15ms per problem, regardless of how long SolveCPU takes.
For the current test workload:
- Small problems (K≤63): SolveCPU ~1-5ms → fully overlapped, pipeline eliminates all wait
- Medium problems (K~127): SolveCPU ~20-50ms → partial overlap
- Large problems (K≥191): SolveCPU ~100-250ms → bottleneck, wait dominates
