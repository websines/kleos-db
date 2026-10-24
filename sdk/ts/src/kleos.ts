/**
 * The {@link Kleos} client: one typed API over the embedded or the remote
 * backend.
 *
 * @module
 */
import { encodeAttachments, KleosError, snakeKeys } from './backend.js';
import type { Backend } from './backend.js';
import type {
  BranchInfo,
  ContextBlock,
  ContextOptions,
  EntityProfile,
  FlowWorker,
  FlowWorkerOptions,
  ForgetOptions,
  ForgetResult,
  ForkResult,
  InitOptions,
  Json,
  MemoryConfig,
  Micros,
  ModelSpec,
  NamedRows,
  RecallHit,
  RecallOptions,
  ReflectResult,
  RememberOptions,
  RememberResult,
} from './types.js';

type Opts = Record<string, unknown> | undefined;

/** kleos: agent memory, documents, ontology, branches and durable workflows. */
export class Kleos {
  /** The transport: embedded (native) or remote (HTTP). */
  readonly backend: Backend;

  constructor(backend: Backend) {
    this.backend = backend;
  }

  /** `embedded` or `remote`. */
  get kind(): 'embedded' | 'remote' {
    return this.backend.kind;
  }

  private mem(op: string, extra: Record<string, unknown>, opts: Opts): Promise<any> {
    return this.backend.memory(op, encodeAttachments({ ...snakeKeys(opts), ...extra }));
  }

  // ---- memory -------------------------------------------------------------

  /** Create (or read) the store's memory configuration. Idempotent. */
  init(opts?: InitOptions): Promise<MemoryConfig> {
    return this.mem('init', {}, opts);
  }

  /**
   * Store a statement. A text of several sentences becomes one memory per
   * claim; a claim that replaces an older one supersedes it (the older one
   * stays readable with `asOf`). Attach documents with `attachments`.
   */
  remember(text: string, opts?: RememberOptions): Promise<RememberResult> {
    return this.mem('remember', { text }, opts);
  }

  /** Memories relevant to `query`, best first. */
  recall(query: string, opts?: RecallOptions): Promise<RecallHit[]> {
    return this.mem('recall', { query, format: 'list' }, opts);
  }

  /** The token-budgeted context block for `query`, ready to paste into a prompt. */
  recallContext(query: string, opts?: ContextOptions): Promise<ContextBlock> {
    return this.mem('recall', { query, format: 'context' }, opts);
  }

  /**
   * Forget a memory by id, or a filter (`ids`, `kind`, `entity`, `expired`,
   * `all`). `mode: 'erase'` deletes every version (GDPR).
   */
  forget(id?: string | null, opts?: ForgetOptions): Promise<ForgetResult> {
    return this.mem('forget', id ? { id } : {}, opts);
  }

  /** Pending decisions, recent changes and contradiction candidates of a namespace. */
  reflect(ns?: string, opts?: { since?: Micros; limit?: number; [k: string]: unknown }): Promise<ReflectResult> {
    return this.mem('reflect', ns ? { ns } : {}, opts);
  }

  /** Change a memory's metadata: `importance`, `ttl`/`expires`, `noTtl`, `archived`. */
  set(id: string | null, opts?: Record<string, unknown>): Promise<any> {
    return this.mem('set', id ? { id } : {}, opts);
  }

  /** Write buffered access counts. */
  flush(): Promise<{ flushed: number }> {
    return this.mem('flush', {}, undefined);
  }

  /** A typed link `a -[rel]-> b` between memories or entities. */
  link(a: string, b: string, rel: string, opts?: { ns?: string; p?: number }): Promise<any> {
    return this.mem('link', { from: a, to: b, rel }, opts);
  }

  /**
   * What memory knows about an entity: its ontology facts plus a context
   * block of the memories that mention it (the same as the CLI's `profile`
   * and the MCP `entity_profile` tool).
   */
  async profile(name: string, opts?: { ns?: string; budget?: number }): Promise<EntityProfile> {
    const ns = opts?.ns ?? 'default';
    const budget = opts?.budget ?? 600;
    let facts: unknown[] = [];
    try {
      const f = await this.backend.ontology('facts', { ns, s: name });
      if (Array.isArray(f)) facts = f;
    } catch {
      // no ontology in this namespace
    }
    let context: ContextBlock = await this.mem('recall', { query: name, entity: name, format: 'context', budget, ns }, undefined);
    if (!context.ids?.length) context = await this.mem('recall', { query: name, format: 'context', budget, ns }, undefined);
    let text = '';
    if (facts.length) text += `Facts about ${name}:\n` + facts.map((f) => `- ${JSON.stringify(f)}`).join('\n') + '\n\n';
    if (context.ids?.length) text += context.text;
    else if (!text) text = `Nothing is known about '${name}' in namespace '${ns}'.`;
    return { name, ns, facts, context, text };
  }

  // ---- branches -----------------------------------------------------------

  /** Fork namespace `from` into a new branch `to` (instant, copy-on-write). */
  fork(from: string, to: string, opts?: { at?: Micros }): Promise<ForkResult> {
    return this.mem('fork', { from, to }, opts);
  }

  /** Put `ns` back to its state at valid time `to`. Undo with `rewind(ns, result.undo)`. */
  rewind(ns: string, to: Micros, opts?: Record<string, unknown>): Promise<any> {
    return this.mem('rewind', { ns, to }, opts);
  }

  /** Replay branch `from` into `into` (default its parent); the decider settles conflicts. `dryRun` previews. */
  merge(from: string, into?: string, opts?: { dryRun?: boolean; keepTime?: boolean; forgets?: boolean }): Promise<any> {
    return this.mem('merge', into ? { from, into } : { from }, opts);
  }

  /** What changes going from namespace `a` to `b`. */
  diff(a: string, b: string, opts?: { at?: Micros }): Promise<any> {
    return this.mem('diff', { a, b }, opts);
  }

  /** Drop a branch's own rows. `cascade` drops its children too. */
  discard(ns: string, opts?: { cascade?: boolean }): Promise<any> {
    return this.mem('discard', { ns }, opts);
  }

  /** List branches. */
  branches(opts?: { ns?: string; all?: boolean }): Promise<BranchInfo[]> {
    return this.mem('branches', {}, opts);
  }

  /** Move the store's vectors to another enabled embedder (a durable, resumable workflow). */
  reembed(embedder: string, opts?: { batch?: number; wait?: boolean; queue?: string }): Promise<any> {
    return this.mem('reembed', { embedder }, opts);
  }

  /** Any memory op by name (escape hatch for ops this client has no method for). */
  memory(op: string, args?: Record<string, unknown>): Promise<any> {
    return this.mem(op, {}, args);
  }

  // ---- ontology, documents, workflows ------------------------------------

  /**
   * An ontology op: `list`, `get`, `define`, `import`, `export`, `activate`,
   * `facts`, `infer`, `validate`, `type`, `learn`, `history`, `undo`, ...
   */
  ontology(op: string, args?: Record<string, unknown>): Promise<any> {
    return this.backend.ontology(op, snakeKeys(args));
  }

  /** A document admin op: `get`, `list`, `delete`, `reingest`. Documents themselves go through `remember`. */
  docs(op: string, args?: Record<string, unknown>): Promise<any> {
    return this.backend.doc(op, snakeKeys(args));
  }

  /** A durable-workflow op: `define`, `start`, `signal`, `cancel`, `status`, `list`, `history`, `schedule`, ... */
  flow(op: string, args?: Record<string, unknown>): Promise<any> {
    return this.backend.flow(op, args ?? {});
  }

  /**
   * Start an in-process worker that runs built-in workflows (background
   * document ingestion, schedules). Embedded only.
   */
  flowWorker(opts?: FlowWorkerOptions): FlowWorker {
    if (!this.backend.flowWorker) {
      throw new KleosError('flowWorker runs in-process: use an embedded store, or start the server with --flow-worker');
    }
    return this.backend.flowWorker(opts ?? {});
  }

  // ---- queries and models -------------------------------------------------

  /** Run a CozoScript query. */
  run(script: string, params?: Record<string, Json>, opts?: { immutable?: boolean }): Promise<NamedRows> {
    return this.backend.run(script, params ?? {}, opts?.immutable ?? false);
  }

  /** Turn an embedder on (`embeddinggemma` downloads on first use). Process-wide when embedded. */
  enableEmbedder(name: string, spec: ModelSpec = {}): Promise<void> {
    return this.backend.enableModel('embedder', name, spec);
  }
  disableEmbedder(name: string): Promise<boolean> {
    return this.backend.disableModel('embedder', name);
  }
  /** Turn a decider on (`nano`: the built-in decider-nano). */
  enableDecider(name: string, spec: ModelSpec = {}): Promise<void> {
    return this.backend.enableModel('decider', name, spec);
  }
  disableDecider(name: string): Promise<boolean> {
    return this.backend.disableModel('decider', name);
  }
  /** Turn an entity extractor on (`gliner`). */
  enableExtractor(name: string, spec: ModelSpec = {}): Promise<void> {
    return this.backend.enableModel('extractor', name, spec);
  }
  disableExtractor(name: string): Promise<boolean> {
    return this.backend.disableModel('extractor', name);
  }
  /** Enabled embedders, as rows. */
  embedders(): Promise<NamedRows> {
    return this.run('::embedder list', {}, { immutable: true });
  }
  deciders(): Promise<NamedRows> {
    return this.run('::decider list', {}, { immutable: true });
  }
  extractors(): Promise<NamedRows> {
    return this.run('::extractor list', {}, { immutable: true });
  }

  /** Close the store (embedded) or release the client (remote). */
  close(): Promise<void> {
    return this.backend.close();
  }
}
