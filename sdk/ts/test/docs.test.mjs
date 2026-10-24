// Code blocks in docs/sdk/typescript marked `<!-- from examples/ts/x.ts -->`
// must appear, line for line and in order, in that example file.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../../..');
const docs = join(root, 'docs/sdk/typescript');

function mdFiles(d) {
  return readdirSync(d).flatMap((f) => {
    const p = join(d, f);
    return statSync(p).isDirectory() ? mdFiles(p) : f.endsWith('.md') ? [p] : [];
  });
}

const files = existsSync(docs) ? mdFiles(docs) : [];

test('docs exist', () => {
  assert.ok(files.length >= 8, `found ${files.length} doc files`);
});

for (const file of files) {
  const md = readFileSync(file, 'utf8');
  const re = /<!-- from (examples\/ts\/[\w.-]+\.ts) -->\s*```ts\n([\s\S]*?)```/g;
  let m;
  while ((m = re.exec(md))) {
    const [, src, block] = m;
    test(`${file.slice(docs.length + 1)} snippet matches ${src}`, () => {
      const example = readFileSync(join(root, src), 'utf8').split('\n').map((l) => l.trimEnd());
      let at = 0;
      for (const line of block.split('\n').map((l) => l.trimEnd()).filter((l) => l.trim() && l.trim() !== '// ...')) {
        const i = example.indexOf(line, at);
        assert.ok(i >= 0, `line not found (in order) in ${src}: ${line}`);
        at = i + 1;
      }
    });
  }
}
