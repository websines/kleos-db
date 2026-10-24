/**
 * Remote backend: a `kleos server` over HTTP. Pure TypeScript with `fetch`,
 * no native code, so it runs in Node, Bun, Deno, browsers, serverless and
 * edge runtimes.
 *
 * @module
 */
import { enableScript, KleosError, scriptLiteral } from './backend.js';
import type { Backend, ModelKind } from './backend.js';
import type { Json, ModelSpec, NamedRows } from './types.js';

/** Options of {@link HttpBackend}. */
export interface ConnectOptions {
  /** Server base URL, e.g. `http://127.0.0.1:9070`. */
  url: string;
  /** Bearer token (`Authorization: Bearer ...`), or the server's admin secret. */
  token?: string;
  /** A `fetch` implementation; defaults to the global one. */
  fetch?: typeof fetch;
  /** Extra headers on every request. */
  headers?: Record<string, string>;
  /** Per-request timeout in ms. Default none. */
  timeoutMs?: number;
}

/** The remote backend. */
export class HttpBackend implements Backend {
  readonly kind = 'remote' as const;
  readonly url: string;
  private token?: string;
  private fetchImpl: typeof fetch;
  private headers: Record<string, string>;
  private timeoutMs?: number;

  constructor(opts: ConnectOptions) {
    if (!opts?.url) throw new KleosError('connect: url is required');
    this.url = opts.url.replace(/\/+$/, '');
    this.token = opts.token;
    const f = opts.fetch ?? (globalThis as any).fetch;
    if (!f) throw new KleosError('connect: no fetch available; pass options.fetch');
    this.fetchImpl = f.bind(globalThis);
    this.headers = opts.headers ?? {};
    this.timeoutMs = opts.timeoutMs;
  }

  private async post(path: string, body: unknown, op: string): Promise<any> {
    const headers: Record<string, string> = { 'content-type': 'application/json', accept: 'application/json', ...this.headers };
    if (this.token) headers.authorization = `Bearer ${this.token}`;
    const ctrl = this.timeoutMs ? new AbortController() : undefined;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), this.timeoutMs) : undefined;
    let res: Response;
    try {
      res = await this.fetchImpl(this.url + path, { method: 'POST', headers, body: JSON.stringify(body), signal: ctrl?.signal });
    } catch (e) {
      throw new KleosError(`${op}: request failed: ${(e as Error).message}`, { op, cause: e });
    } finally {
      if (timer) clearTimeout(timer);
    }
    const text = await res.text();
    let j: any;
    try {
      j = text ? JSON.parse(text) : {};
    } catch {
      throw new KleosError(`${op}: HTTP ${res.status}: ${text.slice(0, 200)}`, { op, status: res.status });
    }
    if (!res.ok || j.ok === false) {
      const msg = j.message ?? j.display ?? `HTTP ${res.status}`;
      throw new KleosError(msg, { op, status: res.status });
    }
    return j;
  }

  async memory(op: string, args: Record<string, unknown>) {
    return (await this.post(`/memory/${op}`, args, `memory.${op}`)).result;
  }
  async ontology(op: string, args: Record<string, unknown>) {
    return (await this.post(`/ontology/${op}`, args, `ontology.${op}`)).result;
  }
  async flow(op: string, args: Record<string, unknown>) {
    return (await this.post(`/flow/${op}`, args, `flow.${op}`)).result;
  }
  async doc(op: string, args: Record<string, unknown>) {
    return (await this.post(`/doc/${op}`, args, `doc.${op}`)).result;
  }
  async run(script: string, params: Record<string, Json>, immutable: boolean): Promise<NamedRows> {
    const j = await this.post('/text-query', { script, params, immutable }, 'run');
    const { ok: _ok, took: _took, ...rows } = j;
    return rows as NamedRows;
  }
  async enableModel(kind: ModelKind, name: string, spec: ModelSpec): Promise<void> {
    await this.run(enableScript(kind, name, spec), {}, false);
  }
  async disableModel(kind: ModelKind, name: string): Promise<boolean> {
    const r = await this.run(`::${kind} disable ${scriptLiteral(name)}`, {}, false);
    const v = r.rows?.[0]?.[r.headers.indexOf('disabled')] ?? r.rows?.[0]?.[0];
    return v === true || v === 'true' || v === 'OK';
  }
  async close(): Promise<void> {
    // stateless: nothing to release
  }
}
