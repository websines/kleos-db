# Experimenting in a branch, then merging

A branch is a copy of a namespace that costs nothing to make. Let an agent try
something in a branch; keep the result with `merge`, or throw it away with
`discard`.

Script: [`examples/ts/branches.ts`](../../../../examples/ts/branches.ts).

## Fork and work in the branch

<!-- from examples/ts/branches.ts -->
```ts
await kleos.fork('plan', 'plan-try');
await kleos.remember('We will add a beta program in May.', { ns: 'plan-try' });
console.log('main sees beta:', (await kleos.recallContext('beta', { ns: 'plan' })).text.includes('beta'));
console.log('branch sees beta:', (await kleos.recallContext('beta', { ns: 'plan-try' })).text.includes('beta'));
```

The branch reads its parent's memories as they were at the fork; its own
changes stay in the branch. Branches of branches work too.

## Preview and merge

<!-- from examples/ts/branches.ts -->
```ts
const preview = await kleos.merge('plan-try', 'plan', { dryRun: true });
console.log('dry run would replay:', preview.statements);
await kleos.merge('plan-try', 'plan');
```

`merge` replays the branch's own statements into the target through
`remember`, so the decider settles conflicts as it would for any new fact: a
branch that learned "moved to Lisbon" supersedes the target's "lives in
Paris". `dryRun` is an exact preview, and `diff(a, b)` shows what differs.

## Rewind and discard

<!-- from examples/ts/branches.ts -->
```ts
const rw = await kleos.rewind('plan', before);
console.log('rewound, retracted:', rw.retracted.length);
await kleos.rewind('plan', rw.undo); // and back again
```

`rewind(ns, t)` puts a namespace back to its state at `t`. It only appends, so
it can be undone with the returned `undo` time. `discard(ns)` drops a branch's
own rows.
