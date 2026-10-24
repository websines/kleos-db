/**
 * Types shared by the embedded (native) and remote (HTTP) backends.
 *
 * Results are returned exactly as kleos produces them, so field names are
 * snake_case (`valid_from`, `pending_superseded_by`). Options accept either
 * camelCase or snake_case keys; camelCase keys are converted.
 *
 * Times are microseconds since the Unix epoch (`Date.now() * 1000`).
 *
 * @module
 */

/** Microseconds since the Unix epoch. */
export type Micros = number;

/** Any JSON value. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/** Who said a statement. Assistant and other memories are recalled but never decided. */
export type Role = 'user' | 'assistant' | 'other';

/** Scheduling lane for model work: interactive calls go first, bulk work never starves. */
export type Priority = 'interactive' | 'bulk';

/** A document passed to `remember`: a path, raw bytes or base64, or a URL. */
export interface Attachment {
  path?: string;
  /** Raw bytes; sent as base64. */
  bytes?: Uint8Array;
  data_b64?: string;
  url?: string;
  mime?: string;
  name?: string;
}

/** Options of {@link Kleos.init}. */
export interface InitOptions {
  /** Default embedder of the store (e.g. `gemma`). Needs an enabled embedder. */
  embedder?: string;
  /** Default decider (e.g. `nano`). */
  decider?: string;
  /** Entity extractor (e.g. `gliner`). */
  extractor?: string;
  /** Ontology for the default namespace: `personal` (the default) or `none`. */
  ontology?: string;
  [key: string]: unknown;
}

/** The store's memory configuration, as returned by `init`. */
export interface MemoryConfig {
  version: number;
  embedder: string | null;
  decider: string | null;
  dim: number | null;
  [key: string]: unknown;
}

/** Options of {@link Kleos.remember}. */
export interface RememberOptions {
  ns?: string;
  kind?: string;
  source?: string;
  /** In [0, 1]. */
  importance?: number;
  /** Names the statement is about; include the subject (`user`, or `Anna` for "my sister Anna …"). */
  entities?: string[];
  /** Valid-from, µs. Default now. */
  at?: Micros;
  /** Expiry relative to `at`, µs. */
  ttl?: Micros;
  /** Absolute expiry, µs. */
  expires?: Micros;
  /** Candidates compared with the new statement. Default 5. */
  k?: number;
  decider?: string;
  embedder?: string;
  /** `auto` (default): one memory per claim; `none`: one memory. */
  split?: 'auto' | 'none';
  minSimilarity?: number;
  role?: Role;
  priority?: Priority;
  /** Documents (PDF, images, Markdown, HTML, DOCX). Needs the `docs` build for embedded use. */
  attachments?: (string | Attachment)[];
  /** A document path instead of text (bytes go in `attachments`). */
  file?: string;
  /** Turn automatic entity extraction off for this call. */
  extract?: boolean;
  [key: string]: unknown;
}

/** One decision `remember` made about an older memory. */
export interface Decision {
  template?: string;
  candidate?: string;
  candidate_text?: string;
  claim?: string;
  action?: string;
  band?: string;
  p?: number;
  decided_by?: string;
  [key: string]: unknown;
}

/** A claim's ontology typing. */
export interface ClaimType {
  claim: string;
  cls: string;
  prop?: string | null;
  subject?: string | null;
  score?: number;
  by?: string;
  [key: string]: unknown;
}

/** Result of {@link Kleos.remember}. */
export interface RememberResult {
  /** `added`, `duplicate`, or `documents` (only files were given). */
  action: 'added' | 'duplicate' | 'documents' | string;
  id: string | null;
  ns: string;
  superseded: string[];
  duplicate_of: string | null;
  /** Decisions left pending for review. */
  tentative: Decision[];
  decisions: Decision[];
  /** Candidates whose supersede question was skipped (different subjects). */
  gated: unknown[];
  entities: string[];
  entity_links: unknown[];
  embedded: boolean;
  decider: string | null;
  claims?: { id: string; text: string; [key: string]: unknown }[];
  types?: ClaimType[];
  /** Documents attached to the call. */
  documents?: DocRef[];
  /** Relation edges from automatic entity extraction. */
  edges?: unknown[];
  extractor?: string | null;
  [key: string]: unknown;
}

/** A document created or found by `remember`. */
export interface DocRef {
  doc: string;
  name?: string;
  status?: string;
  pages?: number;
  chunks?: number;
  flow?: string | null;
  [key: string]: unknown;
}

/** Options of {@link Kleos.recall}. */
export interface RecallOptions {
  ns?: string;
  /** Default 10. */
  k?: number;
  kind?: string;
  entity?: string;
  /** Read the past: valid time, µs. */
  asOf?: Micros;
  vectorWeight?: number;
  /** Leave out memories with a pending supersede. */
  hidePending?: boolean;
  includeArchived?: boolean;
  includeExpired?: boolean;
  /** `['memory']`, `['docs']` or both (default). */
  sources?: ('memory' | 'docs')[];
  /** Entity-graph expansion. */
  expand?: boolean;
  /** Graph modes, where the build supports them. */
  graph?: boolean | string;
  from?: Micros;
  to?: Micros;
  eventFrom?: Micros;
  eventTo?: Micros;
  role?: Role;
  priority?: Priority;
  recordAccess?: boolean;
  [key: string]: unknown;
}

/** Options of {@link Kleos.recallContext}. */
export interface ContextOptions extends RecallOptions {
  /** Token budget. Default 2000. */
  budget?: number;
  showSource?: boolean;
}

/** One recalled memory. */
export interface RecallHit {
  id: string;
  text: string;
  score: number;
  relevance: number;
  recency: number;
  kind: string;
  source: string | null;
  created: Micros;
  importance: number | null;
  valid_from: Micros;
  valid_to: Micros | null;
  /** Newer memories that may replace this one (pending review). */
  pending_superseded_by: string[];
  access_count: number;
  last_access: Micros | null;
  expires: Micros | null;
  archived: boolean;
  /** The whole statement, when the memory is one claim of a longer text. */
  context?: string;
  /** The memory's own text, when it is one claim. */
  claim?: string;
  events?: unknown[];
  role?: Role;
  expanded?: boolean;
  /** Document provenance of a chunk hit. */
  doc?: { doc: string; name?: string; page?: number; page_end?: number; heading?: string };
  [key: string]: unknown;
}

/** The ready-to-paste context block. */
export interface ContextBlock {
  text: string;
  tokens: number;
  budget: number;
  ids: string[];
  omitted: number;
  [key: string]: unknown;
}

/** Options of {@link Kleos.forget}. */
export interface ForgetOptions {
  ns?: string;
  ids?: string[];
  kind?: string;
  entity?: string;
  expired?: boolean;
  all?: boolean;
  /** `retract` (default) ends validity; `erase` deletes every version (GDPR); `archive`. */
  mode?: 'retract' | 'erase' | 'archive';
  [key: string]: unknown;
}

/** Result of {@link Kleos.forget}. */
export interface ForgetResult {
  ids: string[];
  mode: string;
  [key: string]: unknown;
}

/** Result of {@link Kleos.reflect}. */
export interface ReflectResult {
  ns: string;
  pending: unknown[];
  changes: { at: Micros; change: string; id: string; kind: string; text: string }[];
  contradictions: unknown[];
  [key: string]: unknown;
}

/** Result of {@link Kleos.fork}. */
export interface ForkResult {
  ns: string;
  parent: string;
  at: Micros;
  depth: number;
  lineage: [string, Micros | null][];
  [key: string]: unknown;
}

/** A memory branch. */
export interface BranchInfo {
  ns: string;
  parent: string | null;
  at: Micros;
  created: Micros;
  depth: number;
  own: number;
  status: string;
  [key: string]: unknown;
}

/** Result of a CozoScript query. */
export interface NamedRows {
  headers: string[];
  rows: Json[][];
  next?: NamedRows | null;
  [key: string]: unknown;
}

/** A built-in model specification. */
export interface ModelSpec {
  /** Model name or Hugging Face repo, e.g. `embeddinggemma`, `nano`, `gliner`. */
  model?: string;
  /** Local weights directory (deciders and extractors). */
  path?: string;
  /** `f32`, `f16` or `q8`. */
  precision?: string;
  /** `auto`, `cpu` or `metal`. */
  device?: string;
  /** Output width (embedders). */
  dim?: number;
}

/** Options of a native in-process workflow worker. */
export interface FlowWorkerOptions {
  queues?: string[];
  pollMs?: number;
  leaseMs?: number;
  worker?: string;
}

/** A running in-process workflow worker. */
export interface FlowWorker {
  wake(): void;
  stop(): void;
}

/** Result of {@link Kleos.profile}. */
export interface EntityProfile {
  name: string;
  ns: string;
  /** Ontology facts with the entity as subject. */
  facts: unknown[];
  /** Context block of the memories that mention it. */
  context: ContextBlock;
  /** Facts and context as one text, ready for a prompt. */
  text: string;
}
