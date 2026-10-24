/**
 * The transport under {@link Kleos}: a native store in this process, or a
 * remote kleos server over HTTP. Both speak the same JSON ops.
 *
 * @module
 */
import type { FlowWorker, FlowWorkerOptions, Json, ModelSpec, NamedRows } from './types.js';

/** Error thrown by kleos (bad arguments, a closed store, a failed op). */
export class KleosError extends Error {
  /** HTTP status, for remote calls. */
  readonly status?: number;
  /** The op that failed, e.g. `memory.remember`. */
  readonly op?: string;
  constructor(message: string, opts: { status?: number; op?: string; cause?: unknown } = {}) {
    super(message, opts.cause !== undefined ? { cause: opts.cause } : undefined);
    this.name = 'KleosError';
    this.status = opts.status;
    this.op = opts.op;
  }
}

/** The model registries kleos keeps per process (embedders, deciders, extractors). */
export type ModelKind = 'embedder' | 'decider' | 'extractor';

/** What every backend implements. */
export interface Backend {
  /** `embedded` or `remote`. */
  readonly kind: 'embedded' | 'remote';
  memory(op: string, args: Record<string, unknown>): Promise<any>;
  ontology(op: string, args: Record<string, unknown>): Promise<any>;
  flow(op: string, args: Record<string, unknown>): Promise<any>;
  doc(op: string, args: Record<string, unknown>): Promise<any>;
  run(script: string, params: Record<string, Json>, immutable: boolean): Promise<NamedRows>;
  enableModel(kind: ModelKind, name: string, spec: ModelSpec): Promise<void>;
  disableModel(kind: ModelKind, name: string): Promise<boolean>;
  /** Start an in-process workflow worker. Embedded only. */
  flowWorker?(opts: FlowWorkerOptions): FlowWorker;
  close(): Promise<void>;
}

/** Convert top-level camelCase option keys to snake_case; drop undefined. */
export function snakeKeys(opts: Record<string, unknown> | undefined): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (!opts) return out;
  for (const [k, v] of Object.entries(opts)) {
    if (v === undefined) continue;
    out[k.replace(/[A-Z]/g, (c) => '_' + c.toLowerCase())] = v;
  }
  return out;
}

/** Base64 of bytes, in Node (Buffer) or on the web and edge (btoa). */
export function toBase64(bytes: Uint8Array): string {
  const B = (globalThis as any).Buffer;
  if (B) return B.from(bytes).toString('base64');
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

/** Turn attachments with raw bytes into base64 so they travel as JSON. */
export function encodeAttachments(args: Record<string, unknown>): Record<string, unknown> {
  const conv = (a: unknown): unknown => {
    if (a && typeof a === 'object' && (a as any).bytes instanceof Uint8Array) {
      const { bytes, ...rest } = a as any;
      return { ...rest, data_b64: toBase64(bytes) };
    }
    return a;
  };
  if (Array.isArray(args.attachments)) args.attachments = args.attachments.map(conv);
  if (args.file !== undefined) args.file = conv(args.file);
  return args;
}

/** A CozoScript string literal (JSON escaping is valid CozoScript). */
export function scriptLiteral(v: unknown): string {
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  return JSON.stringify(String(v));
}

/** `::embedder enable name {model: "...", ...}` and friends. */
export function enableScript(kind: ModelKind, name: string, spec: ModelSpec): string {
  const fields = Object.entries(spec)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => `${k}: ${scriptLiteral(v)}`);
  const body = fields.length ? ` {${fields.join(', ')}}` : '';
  return `::${kind} enable ${scriptLiteral(name)}${body}`;
}
