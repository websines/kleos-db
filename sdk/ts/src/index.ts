/**
 * kleos for TypeScript.
 *
 * ```ts
 * import { open, connect } from 'kleos';
 * const kleos = open({ path: 'memory.db' });          // embedded (native)
 * // const kleos = connect({ url: 'http://127.0.0.1:9070', token }); // remote
 * await kleos.init();
 * await kleos.remember('I moved to Lisbon in March.', { ns: 'user-42' });
 * const ctx = await kleos.recallContext('Where do I live?', { ns: 'user-42' });
 * ```
 *
 * @module
 */
import { Kleos } from './kleos.js';
import { NativeBackend } from './native.js';
import type { OpenOptions } from './native.js';

export * from './remote.js';
export { NativeBackend, loadNative, nativeFeatures, nativeCandidates } from './native.js';
export type { OpenOptions } from './native.js';

/** Open an embedded store in this process (needs the native module). */
export function open(opts: OpenOptions = {}): Kleos {
  return new Kleos(new NativeBackend(opts));
}
