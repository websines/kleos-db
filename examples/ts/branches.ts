// Experimenting in a branch, then merging: a fork is instant and
// copy-on-write; merge replays the branch's statements through `remember`,
// so conflicts are settled the same way as normal writes.
import { open } from 'kleos';
import { join } from 'node:path';
import { tempDir } from './_util.ts';

const kleos = open({ path: join(tempDir(), 'branches.db') });
await kleos.init();

await kleos.remember('The launch is planned for June.', { ns: 'plan' });
const before = Date.now() * 1000;

// try an idea without touching the main namespace
await kleos.fork('plan', 'plan-try');
await kleos.remember('We will add a beta program in May.', { ns: 'plan-try' });
console.log('main sees beta:', (await kleos.recallContext('beta', { ns: 'plan' })).text.includes('beta'));
console.log('branch sees beta:', (await kleos.recallContext('beta', { ns: 'plan-try' })).text.includes('beta'));

// preview, then merge
const preview = await kleos.merge('plan-try', 'plan', { dryRun: true });
console.log('dry run would replay:', preview.statements);
await kleos.merge('plan-try', 'plan');
console.log('after merge main sees beta:', (await kleos.recallContext('beta', { ns: 'plan' })).text.includes('beta'));

// rewind main to before the experiment; the result carries an undo point
const rw = await kleos.rewind('plan', before);
console.log('rewound, retracted:', rw.retracted.length);
await kleos.rewind('plan', rw.undo); // and back again

// throw the branch away
await kleos.discard('plan-try');
const branches = await kleos.branches({ all: true });
console.log('branches:', branches.filter((b) => !b.ns.includes('~')).map((b) => `${b.ns}:${b.status}`));

await kleos.close();
console.log('ok: branches');
