// A chat agent with per-user memory: each user gets a namespace, every turn
// is remembered, and each answer is grounded in a recalled context block.
import { open } from 'kleos';
import type { Kleos } from 'kleos';
import { join } from 'node:path';
import { tempDir } from './_util.ts';

const kleos = open({ path: join(tempDir(), 'chat.db') });
await kleos.init();

// Stand-in for your LLM call: it just shows what the model would see.
async function llm(prompt: string): Promise<string> {
  return `(model reply to a ${prompt.length}-character prompt)`;
}

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

await turn(kleos, 'ana', 'I am vegetarian and I live in Porto.');
await turn(kleos, 'ben', 'I love steak houses.');
await turn(kleos, 'ana', 'Can you suggest a restaurant for tonight?');

// each user's memory is separate (without an embedder recall matches words;
// enable one, see models.ts, and "food preferences" finds "vegetarian" too)
const ana = await kleos.recallContext('vegetarian food', { ns: 'user:ana' });
const ben = await kleos.recallContext('vegetarian food', { ns: 'user:ben' });
console.log('ana knows vegetarian:', ana.text.includes('vegetarian'));
console.log('ben knows vegetarian:', ben.text.includes('vegetarian'));

await kleos.close();
console.log('ok: chat-agent');
