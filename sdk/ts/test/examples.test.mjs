// Every example under examples/ts runs and ends with "ok: <name>", so the
// guides tied to them can't rot. Needs Node >= 22.6 (TypeScript type
// stripping) and `npm install` in examples/ts (done here if missing).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { KLEOS_BIN } from './helpers.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const dir = resolve(here, '../../../examples/ts');
const [major, minor] = process.versions.node.split('.').map(Number);
const strips = major > 23 || (major === 23 && minor >= 6) || (major === 22 && minor >= 6);
const nodeArgs = major >= 23 && (major > 23 || minor >= 6) ? [] : ['--experimental-strip-types'];
const needsBin = new Set(['mcp', 'remote']);

if (strips && !existsSync(join(dir, 'node_modules', 'kleos'))) {
  execFileSync('npm', ['install', '--no-audit', '--no-fund'], { cwd: dir, stdio: 'ignore' });
}

for (const f of readdirSync(dir).filter((f) => f.endsWith('.ts') && !f.startsWith('_')).sort()) {
  const name = f.replace(/\.ts$/, '');
  const skip = !strips ? 'needs Node >= 22.6' : needsBin.has(name) && !KLEOS_BIN ? 'set KLEOS_BIN' : false;
  test(`example ${f}`, { skip }, () => {
    const out = execFileSync(process.execPath, [...nodeArgs, f], { cwd: dir, encoding: 'utf8', env: process.env, timeout: 300000 });
    assert.match(out, new RegExp(`ok: ${name}\\s*$`));
  });
}
