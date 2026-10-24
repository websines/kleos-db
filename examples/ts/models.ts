// Enabling the built-in models. They download on first use (from Hugging
// Face, into HF_HOME) and run locally; KLEOS_MEMORY_BUDGET caps the memory
// they may use together. Needs a native module built with `infer`
// (or `infer-metal` on Apple silicon).
import { open, nativeFeatures } from 'kleos';
import { join } from 'node:path';
import { tempDir } from './_util.ts';

if (!nativeFeatures().some((f) => f.startsWith('infer'))) {
  console.log('this native module was built without `infer`; skipping (build with KLEOS_FEATURES=infer-metal)');
  console.log('ok: models');
  process.exit(0);
}

const kleos = open({ path: join(tempDir(), 'models.db') });
await kleos.enableEmbedder('gemma', { model: 'embeddinggemma', precision: 'q8' }); // EmbeddingGemma 2
await kleos.enableDecider('nano');      // decider-nano: does a new fact replace an old one?
await kleos.enableExtractor('gliner');  // names and relatives, automatically
await kleos.init({ embedder: 'gemma', decider: 'nano', extractor: 'gliner' });

await kleos.remember('I live in Paris.', { ns: 'me' });
const r = await kleos.remember('I moved to Lisbon last month.', { ns: 'me' });
console.log('superseded:', r.superseded.length, 'pending:', r.tentative.length);
console.log((await kleos.recallContext('Where do I live?', { ns: 'me' })).text);

await kleos.close();
console.log('ok: models');
