# Long-running workflows with signals

A workflow survives crashes, restarts and long waits. Use one when an agent's
task spans hours or days, for example waiting for a human approval.

Script: [`examples/ts/workflows.ts`](../../../../examples/ts/workflows.ts).

## Define

A built-in workflow is a list of steps, stored in kleos:

<!-- from examples/ts/workflows.ts -->
```ts
await kleos.flow('define', {
  name: 'refund',
  graph: {
    steps: [
      { name: 'asked', remember: { text: "=concat('Refund requested for order ', get($input, 'order'))", ns: 'ops' } },
      { name: 'decision', wait_signal: { name: 'approve', timeout_ms: 3 * 24 * 3600 * 1000 } },
      { remember: { text: "=concat('Refund for order ', get($input, 'order'), ' approved by ', get($decision, 'by'))", ns: 'ops' } },
      { return: '$decision' },
    ],
  },
});
```

- **Values:** a string `"$name"` is a named step's result, or `$input`. A
  string starting with `=` is a CozoScript expression (`get($input, 'order')`
  reads a field).
- **Steps:** `query`, `remember`, `recall`, `embed`, `decide`, `sleep {ms}`,
  `wait_signal {name, timeout_ms}`, `child`, `set` and `return`.
- **Control flow:** `if`, `loop`, `for_each` and `fan_out` (parallel
  branches).
- **Retries:** any step can carry `retry {max_attempts, initial_ms, backoff}`.

See [the workflows spec](../../../specs/durable-workflows.md) for the full
list.

## Run, wait, signal

<!-- from examples/ts/workflows.ts -->
```ts
const worker = kleos.flowWorker({ queues: ['default'], pollMs: 100 });
// ...
await kleos.flow('start', { workflow: 'refund', id: 'refund-1001', input: { order: '1001' } });
// ...
await kleos.flow('signal', { id: 'refund-1001', name: 'approve', payload: { by: 'maria' } });
```

- **The worker:** `flowWorker()` runs workflows in this process. On a server,
  start it with `kleos server --flow-worker default`.
- **Waiting:** the workflow waits for the `approve` signal. The signal can come
  from any process or an HTTP call (`POST /flow/signal`), even days later and
  after restarts.
- **Exactly once:** each step is saved as it finishes. The `remember` steps
  write their memories exactly once, even if the worker crashes and the
  workflow replays.

`flow('status', { id })` shows the state (`waiting`, `completed`, `failed`),
and `flow('history', { id })` every recorded step. `flow('cancel', …)`,
`flow('list', …)` and `flow('schedule', …)` (cron) complete the set.
