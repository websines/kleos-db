# A chat agent with per-user memory

Each user gets a namespace. On every turn the agent recalls a context block
for the message, puts it in the prompt, and remembers both sides of the
conversation.

Script: [`examples/ts/chat-agent.ts`](../../../../examples/ts/chat-agent.ts).

## One turn

<!-- from examples/ts/chat-agent.ts -->
```ts
async function turn(k: Kleos, userId: string, message: string): Promise<string> {
  const ns = `user:${userId}`;
  const memory = await k.recallContext(message, { ns, budget: 800 });
  const prompt = `You are a helpful assistant.\n\n${memory.text}\n\nUser: ${message}`;
  const reply = await llm(prompt);
  // remember both sides; assistant memories are recalled but never decide anything
  await k.remember(message, { ns, role: 'user', entities: ['user'] });
  await k.remember(reply, { ns, role: 'assistant', split: 'none' });
  return reply;
}
```

- **`recallContext`** returns a dated, grouped text block within `budget`
  tokens. Paste it as is; it tells the model which values are current and
  which were replaced.
- **Recall before you remember,** so the current message doesn't recall itself.
- **`role: 'assistant'`** keeps the model's own words searchable without
  letting them replace what the user said.
- **`entities: ['user']`** marks first-person statements as being about the
  user. With the entity extractor on, kleos does this itself.

## Users stay separate

<!-- from examples/ts/chat-agent.ts -->
```ts
const ana = await kleos.recallContext('vegetarian food', { ns: 'user:ana' });
const ben = await kleos.recallContext('vegetarian food', { ns: 'user:ben' });
console.log('ana knows vegetarian:', ana.text.includes('vegetarian'));
console.log('ben knows vegetarian:', ben.text.includes('vegetarian'));
```

Without an embedder, recall matches words. Enable one (see
[Getting started](../getting-started.md#enabling-models)) and "food
preferences" finds "I am vegetarian" too.

## In production

- **Enable the decider** so "I moved to Lisbon" supersedes "I live in Paris".
- **Show the user** what is pending (`reflect(ns)`) if you want them to confirm
  changes.
- **For GDPR deletion,** call `forget(null, { ns, all: true, mode: 'erase' })`.
