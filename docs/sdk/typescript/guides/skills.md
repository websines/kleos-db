# Skills: procedural memory that measures itself

kleos stores skills (the SKILL.md format) as versions with success rates,
hands an agent only the few that fit its task, rolls back versions that do
worse, and can learn and optimize skills behind a verification gate. Spec:
[docs/specs/skills.md](../../../specs/skills.md).

## Load a library

```ts
import { open } from 'kleos';
const kleos = open({ path: 'agent.db' });
await kleos.enableEmbedder('gemma', { model: 'embeddinggemma' }); // optional: lexical search without it
await kleos.skill('init', { embedder: 'gemma' });
await kleos.skill('import', { path: './skills', activate: true });   // folders or a Claude Code plugin
```

Imported skills that the security scanner flags stay quarantined and are never
served. `skill('export', { path, format: 'plugin' })` writes them back, byte
for byte.

## Before a task: find the few that fit

```ts
const found = await kleos.findSkills('rotate the staging database credentials', { k: 3, disclosure: 'auto' });
for (const s of found.skills) console.log(s.rank, s.name, s.score.toFixed(2), s.stats.success_rate, s.uri);
const top = found.skills[0];
const instructions = top?.body ?? (await kleos.skill('get', { name: top.name })).body;
```

`findSkills` searches descriptions and bodies (BM25 plus embeddings), ranks
with a router that learns from outcomes, and stays within `budget` tokens
(default 2000). `disclosure: 'body'` includes the top skill's instructions.

## After: report the outcome

```ts
await kleos.recordSkillOutcome({ name: top.name }, { success: true, route: found.route, latencyMs: 8200, tokens: 3100 });
```

Pass `route`: it trains the router. A version whose success rate falls
clearly below its predecessor's (Bayesian test, 20 outcomes each by
default) is rolled back automatically.

Workflows record outcomes themselves: a step `{ name: 'load', skill: { op:
'use', name: 'rotate-db-creds' } }` links the workflow to the skill, and
its completion or failure becomes the outcome.

## Versions and undo

```ts
await kleos.skill('put', { skillMd: newText, activate: true });     // a new version supersedes
await kleos.skill('history', { name: 'rotate-db-creds' });          // versions, heads, log
await kleos.skill('rollback', { name: 'rotate-db-creds' });         // previous live version
await kleos.skill('undo', { name: 'rotate-db-creds' });             // one step back
await kleos.skill('asOf', { name: 'rotate-db-creds', at: someMicros }); // what was live then
```

## Learn and optimize (needs an LLM endpoint)

```ts
await kleos.skill('config', { learn: { llm: { url: 'https://api.example.com/v1', model: 'my-model', api_key_env: 'LLM_KEY' }, mode: 'review' } });
await kleos.skill('recordTrace', { task, transcript, output, success: true }); // from your agent runs
const run = await kleos.skill('learn', {});            // a durable workflow; scheduled daily once configured
const props = await kleos.skill('proposals', {});      // verified candidates (review mode)
await kleos.skill('evalAdd', { name: 'ticket-ids', cases: [{ task, expected, split: 'holdout' }] });
await kleos.skill('optimize', { name: 'ticket-ids', rounds: 4 }); // bounded edits, held-out gate
```

The key stays in the environment variable; kleos stores only its name.

## Share

```ts
const key = await kleos.skill('keygen', {});                    // keep key.secret_key out of the store
await kleos.skill('sign', { name: 'triage', secretKey: key.secret_key });
await kleos.skill('publish', { name: 'triage' });               // served at GET /skills/registry by `kleos server`
await other.skill('trust', { publicKey: key.public_key });
await other.skill('install', { url: 'http://kleos-a:9070', name: 'triage', activate: true });
await kleos.skill('promote', { name: 'triage', scope: 'agent:bot', to: 'team:support' }); // waits for approval
await kleos.skill('approve', { promotion: id, actor: { human: 'alice' } });
```

## Over MCP

The kleos MCP server implements skills over MCP (SEP-2640): `skills/list`
(a partial, most-useful listing; `{task}` for a task's top-k), `skills/get`,
`skill://` resources and `resources/directory/read`, plus the tools
`find_skills`, `get_skill`, `record_skill_outcome`, `propose_skill` and
`skill_admin`. With `McpClient`: `listSkills({ task })`, `getSkill(uri)`,
`readSkillFile(entry, uri)` (verifies the digest and size) and
`readDirectory(uri)`.
