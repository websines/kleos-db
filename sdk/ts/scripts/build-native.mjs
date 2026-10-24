// Build the native module (kleos-lib-nodejs, napi-rs) and copy it to
// native/kleos.<platform>-<arch>.node, where the SDK looks for it.
//
//   node scripts/build-native.mjs                 # release, default features
//   KLEOS_FEATURES=infer-metal,docs node scripts/build-native.mjs
//   KLEOS_PROFILE=debug node scripts/build-native.mjs
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const crate = resolve(here, '../../../kleos-lib-nodejs');
const profile = process.env.KLEOS_PROFILE === 'debug' ? 'debug' : 'release';
const features = (process.env.KLEOS_FEATURES || '').split(',').map((s) => s.trim()).filter(Boolean);
const args = ['build', '--manifest-path', join(crate, 'Cargo.toml')];
if (profile === 'release') args.push('--release');
if (features.length) args.push('--features', features.join(','));
console.log(`cargo ${args.join(' ')}`);
execFileSync('cargo', args, { stdio: 'inherit' });

const targetDir = process.env.CARGO_TARGET_DIR || join(crate, 'target');
const lib = { darwin: 'libkleos_node.dylib', linux: 'libkleos_node.so', win32: 'kleos_node.dll' }[process.platform];
if (!lib) throw new Error(`unsupported platform ${process.platform}`);
const src = join(targetDir, profile, lib);
if (!existsSync(src)) throw new Error(`build output not found: ${src}`);
const out = resolve(here, '../native', `kleos.${process.platform}-${process.arch}.node`);
mkdirSync(dirname(out), { recursive: true });
copyFileSync(src, out);
console.log(`native module: ${out}`);
