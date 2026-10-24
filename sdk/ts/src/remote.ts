/**
 * Remote-only entry point (`kleos/remote`): the HTTP and MCP clients without
 * the native module, for serverless, edge and browser bundles.
 *
 * @module
 */
import { HttpBackend } from './http.js';
import type { ConnectOptions } from './http.js';
import { Kleos } from './kleos.js';

export { Kleos } from './kleos.js';
export { HttpBackend } from './http.js';
export type { ConnectOptions } from './http.js';
export { KleosError } from './backend.js';
export type { Backend, ModelKind } from './backend.js';
export * from './mcp.js';
export type * from './types.js';

/** Connect to a remote `kleos server`. */
export function connect(opts: ConnectOptions): Kleos {
  return new Kleos(new HttpBackend(opts));
}
