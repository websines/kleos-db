/**
 * MCP client for kleos's stateless MCP server (protocol 2026-07-28; see
 * docs/specs/mcp-cli-skills.md). Two transports: Streamable HTTP
 * (`kleos server --mcp`, `POST /mcp`) and stdio (a spawned `kleos mcp`).
 *
 * There is no session: every request carries the protocol version and the
 * client's capabilities in `_meta`. Destructive tools ask for confirmation
 * with a multi-round-trip `input_required` result; pass `onInputRequired` to
 * answer it (or pass `confirm: true` in the tool arguments).
 *
 * @module
 */
import { KleosError } from './backend.js';

/** The protocol version this client speaks. */
export const MCP_PROTOCOL_VERSION = '2026-07-28';

/** A JSON-RPC transport: send one request, get one response. */
export interface McpTransport {
  request(method: string, params: Record<string, unknown>, name?: string): Promise<any>;
  close?(): Promise<void>;
}

/** An `elicitation/create` request inside `input_required`. */
export interface InputRequest {
  method: string;
  params: { mode?: string; message?: string; requestedSchema?: unknown; [k: string]: unknown };
}

/** An answer to one input request. */
export interface InputResponse {
  action: 'accept' | 'decline' | 'cancel';
  content?: Record<string, unknown>;
}

/** Options of {@link McpClient}. */
export interface McpClientOptions {
  transport: McpTransport;
  clientInfo?: { name: string; version: string };
  /** Answer `input_required` requests (e.g. confirmations). Without it, destructive tools need `confirm: true`. */
  onInputRequired?: (requests: Record<string, InputRequest>) => Promise<Record<string, InputResponse>>;
  /** Ask for the Tasks extension (long calls return a task to poll). Default true. */
  tasks?: boolean;
}

/** A tool's result: text content plus the structured JSON result. */
export interface ToolResult {
  content: { type: string; text?: string; [k: string]: unknown }[];
  structuredContent?: any;
  isError?: boolean;
  resultType?: string;
  [k: string]: unknown;
}

/** A task returned by a long call (Tasks extension). */
export interface McpTask {
  taskId: string;
  status: 'working' | 'completed' | 'failed' | 'cancelled' | string;
  pollIntervalMs?: number;
  result?: ToolResult;
  error?: unknown;
  [k: string]: unknown;
}

let nextId = 1;

function rpcBody(method: string, params: Record<string, unknown>) {
  return { jsonrpc: '2.0', id: nextId++, method, params };
}

function unwrap(resp: any, method: string): any {
  if (resp?.error) {
    const e = resp.error;
    throw new KleosError(`MCP ${method}: ${e.message ?? 'error'} (${e.code})`, { op: `mcp.${method}` });
  }
  return resp?.result;
}

/** `Mcp-Name` value: plain when ASCII, else the base64 sentinel. */
function headerName(name: string): string {
  if (/^[\x20-\x7e]*$/.test(name)) return name;
  const B = (globalThis as any).Buffer;
  const b64 = B ? B.from(name, 'utf8').toString('base64') : btoa(unescape(encodeURIComponent(name)));
  return `=?base64?${b64}?=`;
}

/** Streamable HTTP transport (stateless; no session ids). */
export class McpHttpTransport implements McpTransport {
  private fetchImpl: typeof fetch;
  constructor(private opts: { url: string; token?: string; fetch?: typeof fetch; headers?: Record<string, string> }) {
    const f = opts.fetch ?? (globalThis as any).fetch;
    if (!f) throw new KleosError('McpHttpTransport: no fetch available');
    this.fetchImpl = f.bind(globalThis);
  }
  async request(method: string, params: Record<string, unknown>, name?: string): Promise<any> {
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      accept: 'application/json, text/event-stream',
      'mcp-protocol-version': MCP_PROTOCOL_VERSION,
      'mcp-method': method,
      ...this.opts.headers,
    };
    if (name !== undefined) headers['mcp-name'] = headerName(name);
    if (this.opts.token) headers.authorization = `Bearer ${this.opts.token}`;
    const url = this.opts.url.replace(/\/+$/, '').replace(/\/mcp$/, '') + '/mcp';
    const res = await this.fetchImpl(url, { method: 'POST', headers, body: JSON.stringify(rpcBody(method, params)) });
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch {
      throw new KleosError(`MCP ${method}: HTTP ${res.status}: ${text.slice(0, 200)}`, { status: res.status, op: `mcp.${method}` });
    }
  }
}

/**
 * stdio transport: spawns `kleos mcp` (Node only) and speaks
 * newline-delimited JSON-RPC.
 */
export class McpStdioTransport implements McpTransport {
  private child: any;
  private pending = new Map<number, (v: any) => void>();
  private buf = '';
  private ready: Promise<void>;

  /** `command` defaults to `kleos`; `args` e.g. `['mcp', '--db', 'store.db']`. */
  constructor(private opts: { command?: string; args?: string[]; env?: Record<string, string> } = {}) {
    this.ready = this.start();
  }
  private async start() {
    const { spawn } = await import('node:child_process');
    this.child = spawn(this.opts.command ?? 'kleos', this.opts.args ?? ['mcp'], {
      stdio: ['pipe', 'pipe', 'inherit'],
      env: { ...process.env, ...this.opts.env },
    });
    this.child.stdout.setEncoding('utf8');
    this.child.stdout.on('data', (chunk: string) => {
      this.buf += chunk;
      let i: number;
      while ((i = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, i).trim();
        this.buf = this.buf.slice(i + 1);
        if (!line) continue;
        try {
          const msg = JSON.parse(line);
          const r = this.pending.get(msg.id);
          if (r) {
            this.pending.delete(msg.id);
            r(msg);
          }
        } catch {
          // ignore non-JSON lines
        }
      }
    });
  }
  async request(method: string, params: Record<string, unknown>): Promise<any> {
    await this.ready;
    const body = rpcBody(method, params);
    return new Promise((resolve) => {
      this.pending.set(body.id, resolve);
      this.child.stdin.write(JSON.stringify(body) + '\n');
    });
  }
  async close(): Promise<void> {
    await this.ready;
    this.child?.stdin.end();
    this.child?.kill();
  }
}

/** A stateless MCP client for kleos. */
export class McpClient {
  private t: McpTransport;
  private clientInfo: { name: string; version: string };
  private onInputRequired?: McpClientOptions['onInputRequired'];
  private tasks: boolean;

  constructor(opts: McpClientOptions) {
    this.t = opts.transport;
    this.clientInfo = opts.clientInfo ?? { name: 'kleos-ts', version: '0.18.0' };
    this.onInputRequired = opts.onInputRequired;
    this.tasks = opts.tasks ?? true;
  }

  private meta() {
    const caps: Record<string, unknown> = {};
    if (this.onInputRequired) caps.elicitation = { form: {} };
    if (this.tasks) caps.extensions = { 'io.modelcontextprotocol/tasks': {} };
    return {
      'io.modelcontextprotocol/protocolVersion': MCP_PROTOCOL_VERSION,
      'io.modelcontextprotocol/clientCapabilities': caps,
      'io.modelcontextprotocol/clientInfo': this.clientInfo,
    };
  }

  private async rpc(method: string, params: Record<string, unknown> = {}, name?: string) {
    const resp = await this.t.request(method, { ...params, _meta: this.meta() }, name);
    return unwrap(resp, method);
  }

  /** `server/discover`: versions, capabilities and instructions. */
  discover(): Promise<any> {
    return this.rpc('server/discover');
  }
  /** The server's tools, in a deterministic order. */
  async listTools(): Promise<any[]> {
    return (await this.rpc('tools/list')).tools;
  }
  async listResources(): Promise<any[]> {
    return (await this.rpc('resources/list')).resources;
  }
  async readResource(uri: string): Promise<any> {
    return this.rpc('resources/read', { uri }, uri);
  }

  /**
   * `skills/list` (SEP-2640). kleos answers with the most useful skills (a
   * partial listing) or, given a `task`, the top-k for it.
   */
  async listSkills(opts: { task?: string; k?: number; cursor?: string } = {}): Promise<{ skills: any[]; nextCursor?: string }> {
    return this.rpc('skills/list', opts as Record<string, unknown>);
  }
  /** `skills/get`: one skill's entry (frontmatter and file digests) by its SKILL.md URI. */
  async getSkill(uri: string): Promise<any> {
    return (await this.rpc('skills/get', { uri })).skill;
  }
  /**
   * Read a skill file and verify it against the entry's digest and size, as
   * SEP-2640 requires of hosts. Throws on a mismatch or an unlisted file.
   */
  async readSkillFile(entry: { resources: { uri: string; digest: string; size: number }[] | 'dynamic' }, uri: string): Promise<string> {
    const listed = entry.resources === 'dynamic' ? undefined : entry.resources.find((r) => r.uri === uri);
    if (entry.resources !== 'dynamic' && !listed) throw new KleosError(`${uri} is not listed in the skill entry`, { op: 'mcp.skills' });
    const r = await this.readResource(uri);
    const text: string = r.contents?.[0]?.text ?? '';
    if (listed) {
      const bytes = new TextEncoder().encode(text);
      const { createHash } = await import('node:crypto');
      const digest = 'sha256:' + createHash('sha256').update(bytes).digest('hex');
      if (bytes.length !== listed.size || digest !== listed.digest) throw new KleosError(`${uri} does not match its digest`, { op: 'mcp.skills' });
    }
    return text;
  }
  /** `resources/directory/read`: the children of a skill directory. */
  async readDirectory(uri: string): Promise<any[]> {
    return (await this.rpc('resources/directory/read', { uri }, uri)).resources;
  }

  /**
   * Call a tool. Handles `input_required` (confirmations) through
   * `onInputRequired`; returns a {@link McpTask} when a long call became a task.
   */
  async callTool(name: string, args: Record<string, unknown> = {}): Promise<ToolResult | McpTask> {
    let params: Record<string, unknown> = { name, arguments: args };
    for (let round = 0; round < 4; round++) {
      const r = await this.rpc('tools/call', params, name);
      if (r?.resultType === 'input_required') {
        if (!this.onInputRequired) {
          throw new KleosError(`tool ${name} needs input (confirmation); pass onInputRequired or confirm: true`, { op: 'mcp.tools/call' });
        }
        const responses = await this.onInputRequired(r.inputRequests ?? {});
        params = { name, arguments: args, inputResponses: responses, requestState: r.requestState };
        continue;
      }
      if (r?.resultType === 'task') return r as McpTask;
      return r as ToolResult;
    }
    throw new KleosError(`tool ${name}: too many input rounds`, { op: 'mcp.tools/call' });
  }

  /** Call a tool and return its structured JSON result; throws on `isError`. */
  async call(name: string, args: Record<string, unknown> = {}): Promise<any> {
    const r = await this.callTool(name, args);
    if ((r as McpTask).taskId) return r;
    const t = r as ToolResult;
    if (t.isError) throw new KleosError(t.content?.map((c) => c.text).join('\n') || `tool ${name} failed`, { op: `mcp.${name}` });
    return t.structuredContent ?? t.content?.[0]?.text;
  }

  /** Poll a task (Tasks extension). */
  getTask(taskId: string): Promise<McpTask> {
    return this.rpc('tasks/get', { taskId });
  }
  cancelTask(taskId: string): Promise<McpTask> {
    return this.rpc('tasks/cancel', { taskId });
  }
  /** Poll until the task finishes. */
  async waitTask(taskId: string, opts: { intervalMs?: number; timeoutMs?: number } = {}): Promise<McpTask> {
    const start = Date.now();
    for (;;) {
      const t = await this.getTask(taskId);
      if (t.status !== 'working') return t;
      if (opts.timeoutMs && Date.now() - start > opts.timeoutMs) throw new KleosError(`task ${taskId} timed out`);
      await new Promise((r) => setTimeout(r, opts.intervalMs ?? t.pollIntervalMs ?? 500));
    }
  }
  async close(): Promise<void> {
    await this.t.close?.();
  }
}

/** Confirm every input request (use only when the user already agreed). */
export const acceptAll = async (reqs: Record<string, InputRequest>): Promise<Record<string, InputResponse>> =>
  Object.fromEntries(Object.keys(reqs).map((k) => [k, { action: 'accept', content: { confirm: true } }])) as Record<string, InputResponse>;
