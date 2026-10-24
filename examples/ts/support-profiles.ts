// A support agent using entity profiles: what memory knows about a customer,
// as one prompt-ready text, plus recall filtered to one entity.
import { open } from 'kleos';
import { join } from 'node:path';
import { tempDir } from './_util.ts';

const kleos = open({ path: join(tempDir(), 'support.db') });
await kleos.init();
const ns = 'support';

await kleos.remember('Acme Corp is on the Enterprise plan.', { ns, entities: ['Acme Corp'] });
await kleos.remember('Acme Corp reported login failures on their SSO setup.', { ns, entities: ['Acme Corp'] });
await kleos.remember('Globex uses the Starter plan.', { ns, entities: ['Globex'] });

// everything about one customer, ready for the prompt
const profile = await kleos.profile('Acme Corp', { ns });
console.log(profile.text);

// recall restricted to memories that mention the entity
const hits = await kleos.recall('plan', { ns, entity: 'Acme Corp' });
console.log('Acme memories:', hits.map((h) => h.text));

// what needs review: pending replacements, possible contradictions, recent changes
const review = await kleos.reflect(ns);
console.log('recent changes:', review.changes.length);

await kleos.close();
console.log('ok: support-profiles');
