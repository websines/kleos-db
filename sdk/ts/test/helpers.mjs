// Test helpers: temp dirs, the kleos binary (KLEOS_BIN) and a local server.
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';

export function tempDir(prefix = 'kleos-ts-test-') {
  return mkdtempSync(join(process.env.KLEOS_TMP ?? tmpdir(), prefix));
}

export const KLEOS_BIN = process.env.KLEOS_BIN;

export async function startServer(extra = []) {
  const port = 20000 + Math.floor(Math.random() * 20000);
  const dir = tempDir('kleos-ts-server-');
  const child = spawn(KLEOS_BIN, ['server', '--mcp', '--insecure-no-auth', '--engine', 'sqlite', '--path', join(dir, 's.db'), '--port', String(port), ...extra], {
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  let err = '';
  child.stderr.on('data', (d) => (err += d));
  const url = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 150; i++) {
    try {
      const r = await fetch(url + '/');
      if (r.status < 500) return { url, stop: () => child.kill() };
    } catch {}
    await new Promise((r) => setTimeout(r, 100));
  }
  child.kill();
  throw new Error('server did not start: ' + err);
}
