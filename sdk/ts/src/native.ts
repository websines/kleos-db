/**
 * Embedded backend: a kleos store opened in this process through the native
 * module (`kleos-lib-nodejs`, napi-rs). Every call runs on the libuv thread
 * pool, so the event loop is never blocked.
 *
 * @module
 */
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { KleosError } from './backend.js';
import type { Backend, ModelKind } from './backend.js';
import type { FlowWorker, FlowWorkerOptions, Json, ModelSpec, NamedRows } from './types.js';

interface NativeDb {
  runScript(script: string, paramsJson: string, immutable: boolean): Promise<string>;
  memoryCall(op: string, argsJson: string): Promise<string>;
  ontologyCall(op: string, argsJson: string): Promise<string>;
  flowCall(op: string, argsJson: string): Promise<string>;
  docCall(op: string, argsJson: string): Promise<string>;
  flowWorker(optsJson: string): FlowWorker;
  close(): boolean;
}

interface NativeModule {
  KleosDb: new (engine: string, path: string, options: string) => NativeDb;
  enableEmbedder(name: string, spec: string): Promise<string>;
  disableEmbedder(name: string): Promise<string>;
  enableDecider(name: string, spec: string): Promise<string>;
  disableDecider(name: string): Promise<string>;
  enableExtractor(name: string, spec: string): Promise<string>;
  disableExtractor(name: string): Promise<string>;
  features(): string[];
}

let cached: NativeModule | undefined;

/** Where the native module is looked for, in order. */
export function nativeCandidates(): string[] {
  const here = dirname(fileURLToPath(import.meta.url));
  const file = `kleos.${process.platform}-${process.arch}.node`;
  const out: string[] = [];
  if (process.env.KLEOS_NATIVE) out.push(resolve(process.env.KLEOS_NATIVE));
  out.push(join(here, '..', 'native', file));
  out.push(join(here, 'native', file));
  return out;
}

/** Load (once) the native module. Throws a {@link KleosError} explaining how to build it. */
export function loadNative(): NativeModule {
  if (cached) return cached;
  const require = createRequire(import.meta.url);
  const tried = nativeCandidates();
  for (const p of tried) {
    if (existsSync(p)) {
      cached = require(p) as NativeModule;
      return cached;
    }
  }
  throw new KleosError(
    `kleos native module not found for ${process.platform}-${process.arch}. Tried:\n  ${tried.join('\n  ')}\n` +
      'Build it with `npm run build:native` in sdk/ts (see docs/sdk/typescript/troubleshooting.md), ' +
      'set KLEOS_NATIVE to a built .node file, or use Kleos.connect() for a remote server.',
  );
}

/** Features the loaded native module was built with (`infer`, `docs`, ...). */
export function nativeFeatures(): string[] {
  return loadNative().features();
}

function parseError(e: unknown, op: string): KleosError {
  const msg = e instanceof Error ? e.message : String(e);
  try {
    const j = JSON.parse(msg);
    if (j && typeof j.message === 'string') return new KleosError(j.message, { op, cause: e });
  } catch {
    // not JSON
  }
  return new KleosError(msg, { op, cause: e });
}

async function call(op: string, p: () => Promise<string>): Promise<any> {
  let out: string;
  try {
    out = await p();
  } catch (e) {
    throw parseError(e, op);
  }
  return JSON.parse(out);
}

/** Options of {@link NativeBackend}. */
export interface OpenOptions {
  /** `mem` (default when no path), `sqlite` (default with a path) or `rocksdb`. */
  engine?: 'mem' | 'sqlite' | 'rocksdb';
  /** Store file or directory. */
  path?: string;
  /** Engine options. */
  options?: Record<string, unknown>;
}

/** The embedded backend. */
export class NativeBackend implements Backend {
  readonly kind = 'embedded' as const;
  private db: NativeDb;
  private native: NativeModule;

  constructor(opts: OpenOptions = {}) {
    this.native = loadNative();
    const engine = opts.engine ?? (opts.path ? 'sqlite' : 'mem');
    try {
      this.db = new this.native.KleosDb(engine, opts.path ?? '', JSON.stringify(opts.options ?? {}));
    } catch (e) {
      throw parseError(e, 'open');
    }
  }

  memory(op: string, args: Record<string, unknown>) {
    return call(`memory.${op}`, () => this.db.memoryCall(op, JSON.stringify(args)));
  }
  ontology(op: string, args: Record<string, unknown>) {
    return call(`ontology.${op}`, () => this.db.ontologyCall(op, JSON.stringify(args)));
  }
  flow(op: string, args: Record<string, unknown>) {
    return call(`flow.${op}`, () => this.db.flowCall(op, JSON.stringify(args)));
  }
  doc(op: string, args: Record<string, unknown>) {
    return call(`doc.${op}`, () => this.db.docCall(op, JSON.stringify(args)));
  }
  run(script: string, params: Record<string, Json>, immutable: boolean): Promise<NamedRows> {
    return call('run', () => this.db.runScript(script, JSON.stringify(params), immutable));
  }
  async enableModel(kind: ModelKind, name: string, spec: ModelSpec): Promise<void> {
    const s = JSON.stringify(spec);
    const n = this.native;
    const f = { embedder: n.enableEmbedder, decider: n.enableDecider, extractor: n.enableExtractor }[kind];
    await call(`${kind}.enable`, () => f(name, s));
  }
  async disableModel(kind: ModelKind, name: string): Promise<boolean> {
    const n = this.native;
    const f = { embedder: n.disableEmbedder, decider: n.disableDecider, extractor: n.disableExtractor }[kind];
    return Boolean(await call(`${kind}.disable`, () => f(name)));
  }
  flowWorker(opts: FlowWorkerOptions = {}): FlowWorker {
    try {
      return this.db.flowWorker(JSON.stringify(opts));
    } catch (e) {
      throw parseError(e, 'flowWorker');
    }
  }
  async close(): Promise<void> {
    this.db.close();
  }
}
