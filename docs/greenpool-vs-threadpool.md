# GreenPool vs ThreadPoolExecutor

## What changed and why

The portfolio analyzer originally used `ThreadPoolExecutor` to fetch sentiment and price data
for multiple tickers in parallel. On EC2 (t2.micro), this caused crashes: eventlet and Python's
OS threads conflict because eventlet monkey-patches the stdlib and expects to own the I/O loop.
The fix was to replace `ThreadPoolExecutor` with eventlet's `GreenPool`.

---

## Preemptive vs cooperative threading

### ThreadPoolExecutor — preemptive (OS threads)

The OS decides when to pause one thread and resume another. Threads run truly in parallel
(on multiple cores) and can be interrupted at any point.

```
Thread 1: ──── AAPL fetch ──────────────────────────── done ──
Thread 2: ──── TSLA fetch ────────────────── done ────────────
Thread 3: ──── MSFT fetch ──── done ──────────────────────────
                ↑
          OS scheduler interrupts and switches freely
```

**Problem with eventlet:** eventlet monkey-patches `socket`, `time`, `threading`, etc. at
startup so it can manage I/O cooperatively. When OS threads step in, they bypass eventlet's
scheduler — unpredictably corrupting its internal state and crashing the server.

```python
# ThreadPoolExecutor — breaks on EC2 with eventlet
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(analyze_ticker, t): t for t in tickers}
    for future in as_completed(futures):
        row = future.result()
        socketio.emit("ticker_result", row)  # ← buffered, fires at end
```

`socketio.emit()` inside a non-eventlet thread doesn't yield control back to the eventlet
hub — so emits get queued and all fire at once after the executor shuts down, not as each
ticker completes.

---

### GreenPool — cooperative (green threads)

Green threads (greenlets) are lightweight coroutines. They only switch at explicit yield
points — I/O calls like `requests.post()` or `socket.recv()`. The eventlet hub is the
scheduler, not the OS.

```
Green 1: ── AAPL fetch start ──[waiting on HF API]──────────────── done ──
Green 2: ────────────────────── TSLA fetch start ──[waiting]── done ───────
Green 3: ───────────────────────────────────────── MSFT fetch start ── done
                                 ↑
                    eventlet switches here (at I/O boundary)
```

**Works with eventlet:** GreenPool is part of eventlet — it uses the same hub, the same
scheduler, the same monkey-patched I/O. No conflict.

```python
# GreenPool — eventlet-safe, concurrent I/O
pool = eventlet.GreenPool(size=5)
rows = pool.imap(analyze_and_collect, tickers)
for row in rows:
    socketio.emit("ticker_result", row)  # ← fires immediately as each completes
    socketio.sleep(0.3)
```

`socketio.emit()` inside the eventlet hub fires immediately and yields control cleanly.
Cards stream to the browser one by one as each ticker finishes.

---

## Side-by-side comparison

| | ThreadPoolExecutor | GreenPool |
|---|---|---|
| Thread type | OS threads | Green threads (coroutines) |
| Scheduling | Preemptive (OS) | Cooperative (eventlet hub) |
| True parallelism | Yes (multi-core) | No (single-core) |
| I/O concurrency | Yes | Yes |
| Eventlet-safe | No — crashes on EC2 | Yes |
| `socketio.emit()` | Queued, fires at end | Fires immediately |
| t2.micro (1 vCPU) | No parallel gain | Same performance |

On a single vCPU machine, true parallelism doesn't help. The bottleneck is I/O wait
(HuggingFace API, yfinance HTTP calls) — and both approaches overlap that wait equally.
GreenPool wins because it does the same work without the crash.

---

## Platform detection

The kqueue implementation in eventlet is buggy on macOS, so the app detects the OS at
startup and selects the right mode:

```python
import platform

if platform.system() == "Linux":
    import eventlet
    eventlet.monkey_patch()
    ASYNC_MODE = "eventlet"   # EC2 — GreenPool, full concurrency
else:
    ASYNC_MODE = "threading"  # Mac — sequential, no crash
```

```python
if ASYNC_MODE == "eventlet":
    pool = eventlet.GreenPool(size=5)
    rows = pool.imap(analyze_and_collect, holdings.keys())
else:
    rows = (analyze_and_collect(t) for t in holdings.keys())  # sequential
```

Same codebase, right behavior on both platforms.

---

## Duplicate card fix

The browser's socket client auto-reconnects on any blip. The `connect` event fires on
every reconnect, which was re-emitting `start_analysis` and running the full analysis
twice — doubling every card. Fixed with a one-shot flag:

```js
let analysisStarted = false;
socket.on('connect', () => {
    if (!analysisStarted) {
        analysisStarted = true;
        socket.emit('start_analysis');
    }
});
```
