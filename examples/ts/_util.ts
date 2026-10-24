// Small helpers shared by the examples: a temporary store path and a
// locally started kleos server for the remote examples.
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';

/** A fresh directory for a throwaway store. KLEOS_TMP overrides the parent. */
export function tempDir(prefix = 'kleos-example-'): string {
  return mkdtempSync(join(process.env.KLEOS_TMP ?? tmpdir(), prefix));
}

/** The kleos binary: KLEOS_BIN, else `kleos` on PATH. */
export function kleosBin(): string {
  return process.env.KLEOS_BIN ?? 'kleos';
}

/** Start `kleos server --mcp` on a free port without auth (loopback only); returns its URL and a stop function. */
export async function startServer(): Promise<{ url: string; stop: () => void }> {
  const port = 20000 + Math.floor(Math.random() * 20000);
  const dir = tempDir('kleos-server-');
  const child = spawn(kleosBin(), ['server', '--mcp', '--insecure-no-auth', '--engine', 'sqlite', '--path', join(dir, 'server.db'), '--port', String(port), '--flow-worker', 'default', '--flow-worker', 'docs'], {
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  const url = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 100; i++) {
    try {
      const r = await fetch(url + '/');
      if (r.status < 500) return { url, stop: () => child.kill() };
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  child.kill();
  throw new Error('kleos server did not start');
}
