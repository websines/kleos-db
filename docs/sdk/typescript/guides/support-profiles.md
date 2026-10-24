# A support agent with entity profiles

Support agents need everything known about one customer, current and dated,
in a few hundred tokens. `profile(name)` gives exactly that: the entity's
ontology facts plus a context block of the memories that mention it.

Script: [`examples/ts/support-profiles.ts`](../../../../examples/ts/support-profiles.ts).

## Store facts about customers

<!-- from examples/ts/support-profiles.ts -->
```ts
await kleos.remember('Acme Corp is on the Enterprise plan.', { ns, entities: ['Acme Corp'] });
await kleos.remember('Acme Corp reported login failures on their SSO setup.', { ns, entities: ['Acme Corp'] });
await kleos.remember('Globex uses the Starter plan.', { ns, entities: ['Globex'] });
```

Naming the entity is optional with the entity extractor enabled: it finds
"Acme Corp" itself.

## Read a profile

<!-- from examples/ts/support-profiles.ts -->
```ts
const profile = await kleos.profile('Acme Corp', { ns });
console.log(profile.text);
```

```
## Acme Corp
- [2026-10-24 (Sat)] Acme Corp is on the Enterprise plan.
- [2026-10-24 (Sat)] Acme Corp reported login failures on their SSO setup.
```

`profile.facts` holds the structured ontology facts about the entity, and
`profile.context` the context block. With the decider enabled, when the
customer upgrades ("Acme Corp moved to the Business plan"), the old plan is
superseded and shows up as "previously".

## Recall about one entity

<!-- from examples/ts/support-profiles.ts -->
```ts
const hits = await kleos.recall('plan', { ns, entity: 'Acme Corp' });
console.log('Acme memories:', hits.map((h) => h.text));
```

`entity` matches every name merged into that entity ("Acme", "Acme Corp").

## Review queue

<!-- from examples/ts/support-profiles.ts -->
```ts
const review = await kleos.reflect(ns);
console.log('recent changes:', review.changes.length);
```

`reflect` lists pending replacements, possible contradictions and recent
changes: a natural queue for a human to check.
