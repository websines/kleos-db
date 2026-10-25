/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Detached skill signatures (docs/specs/skills.md, "Signing"): Ed25519 in
//! the **minisign** format, so they work offline and verify with stock
//! `minisign -Vm manifest.json -P <public key>`.
//!
//! - What is signed: the canonical manifest bytes of a version
//!   (`{"files":[{"digest","path","size"},…]}`), which bind every file's
//!   SHA-256; verification also re-checks each file against its digest.
//! - Signature: minisign's prehashed mode (`ED`): Ed25519 over the BLAKE2b-512
//!   of the manifest, key id, and a global signature over the signature plus
//!   the trusted comment (`timestamp`, skill name, manifest digest). Legacy
//!   `Ed` signatures are accepted on verification.
//! - Public key: minisign's (`base64("Ed" ‖ key id ‖ key)`).
//! - Secret key: `kleos-sk1:base64(key id ‖ seed)`, unencrypted; keep it in an
//!   environment variable or a file (`key_env` / `key_file`), never in the
//!   store. minisign's scrypt-encrypted secret-key files are not read.

use base64::Engine;
use blake2::{Blake2b512, Digest};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use miette::{bail, miette, Result};
use serde_json::{json, Value as JsonValue};

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;
const SK_PREFIX: &str = "kleos-sk1:";

/// A key id as minisign prints it: the 8 bytes as a little-endian u64, hex.
pub fn kid_hex(kid: &[u8; 8]) -> String {
    format!("{:016X}", u64::from_le_bytes(*kid))
}

/// A new key pair: `{kid, public_key, secret_key, public_key_file}`. The
/// secret is returned once and never stored.
pub fn keygen(label: Option<&str>) -> Result<JsonValue> {
    let mut seed = [0u8; 32];
    let mut kid = [0u8; 8];
    use rand::RngCore;
    rand::rngs::OsRng.fill_bytes(&mut seed);
    rand::rngs::OsRng.fill_bytes(&mut kid);
    let sk = SigningKey::from_bytes(&seed);
    let public = public_key_b64(&kid, &sk.verifying_key());
    let mut sec = kid.to_vec();
    sec.extend_from_slice(&seed);
    Ok(json!({
        "kid": kid_hex(&kid),
        "public_key": public,
        "secret_key": format!("{SK_PREFIX}{}", B64.encode(sec)),
        "public_key_file": format!("untrusted comment: minisign public key {}\n{public}\n", kid_hex(&kid)),
        "label": label,
        "note": "store secret_key in an environment variable or file; kleos never keeps it",
    }))
}

fn public_key_b64(kid: &[u8; 8], vk: &VerifyingKey) -> String {
    let mut b = b"Ed".to_vec();
    b.extend_from_slice(kid);
    b.extend_from_slice(vk.as_bytes());
    B64.encode(b)
}

/// A secret key from its `kleos-sk1:` form.
pub fn parse_secret(s: &str) -> Result<([u8; 8], SigningKey)> {
    let raw = s.trim().strip_prefix(SK_PREFIX).ok_or_else(|| miette!("a kleos secret key starts with '{SK_PREFIX}'"))?;
    let b = B64.decode(raw).map_err(|_| miette!("the secret key is not valid base64"))?;
    if b.len() != 40 {
        bail!("the secret key has the wrong length");
    }
    let kid: [u8; 8] = b[..8].try_into().unwrap();
    let seed: [u8; 32] = b[8..].try_into().unwrap();
    Ok((kid, SigningKey::from_bytes(&seed)))
}

/// A minisign public key: the base64 line, or a whole `.pub` file.
pub fn parse_public(s: &str) -> Result<([u8; 8], VerifyingKey)> {
    let line = s.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with("untrusted comment:")).unwrap_or("");
    let b = B64.decode(line).map_err(|_| miette!("the public key is not valid base64"))?;
    if b.len() != 42 || &b[..2] != b"Ed" {
        bail!("not a minisign Ed25519 public key");
    }
    let kid: [u8; 8] = b[2..10].try_into().unwrap();
    let vk = VerifyingKey::from_bytes(&b[10..42].try_into().unwrap()).map_err(|e| miette!("bad public key: {e}"))?;
    Ok((kid, vk))
}

/// The public key (minisign form) of a secret key.
pub fn public_of(secret: &str) -> Result<String> {
    let (kid, sk) = parse_secret(secret)?;
    Ok(public_key_b64(&kid, &sk.verifying_key()))
}

/// A minisign signature file for `msg`.
pub fn sign(msg: &[u8], secret: &str, trusted_comment: &str) -> Result<String> {
    let (kid, sk) = parse_secret(secret)?;
    if trusted_comment.contains('\n') {
        bail!("a trusted comment is one line");
    }
    let sig = sk.sign(&Blake2b512::digest(msg));
    let mut sb = b"ED".to_vec();
    sb.extend_from_slice(&kid);
    sb.extend_from_slice(&sig.to_bytes());
    let mut global_msg = sig.to_bytes().to_vec();
    global_msg.extend_from_slice(trusted_comment.as_bytes());
    let global = sk.sign(&global_msg);
    Ok(format!(
        "untrusted comment: signature from kleos secret key {}\n{}\ntrusted comment: {trusted_comment}\n{}\n",
        kid_hex(&kid),
        B64.encode(sb),
        B64.encode(global.to_bytes())
    ))
}

/// Verify a minisign signature file over `msg` with `public` (minisign
/// form). Returns `(key id, trusted comment)`.
pub fn verify(msg: &[u8], sig_file: &str, public: &str) -> Result<(String, String)> {
    let (pkid, vk) = parse_public(public)?;
    let lines: Vec<&str> = sig_file.lines().collect();
    let sig_line = lines.iter().find(|l| !l.trim().is_empty() && !l.starts_with("untrusted comment:")).ok_or_else(|| miette!("empty signature"))?;
    let pos = lines.iter().position(|l| l == sig_line).unwrap();
    let tc = lines
        .get(pos + 1)
        .and_then(|l| l.strip_prefix("trusted comment: "))
        .ok_or_else(|| miette!("the signature has no trusted comment"))?;
    let global_line = lines.get(pos + 2).ok_or_else(|| miette!("the signature has no global signature"))?;
    let sb = B64.decode(sig_line.trim()).map_err(|_| miette!("the signature is not valid base64"))?;
    if sb.len() != 74 {
        bail!("the signature has the wrong length");
    }
    let alg = &sb[..2];
    let kid: [u8; 8] = sb[2..10].try_into().unwrap();
    if kid != pkid {
        bail!("signed by key {}, not {}", kid_hex(&kid), kid_hex(&pkid));
    }
    let sig = Signature::from_bytes(&sb[10..74].try_into().unwrap());
    match alg {
        b"ED" => vk.verify(&Blake2b512::digest(msg), &sig),
        b"Ed" => vk.verify(msg, &sig),
        _ => bail!("unknown signature algorithm"),
    }
    .map_err(|_| miette!("the signature does not verify"))?;
    let gb = B64.decode(global_line.trim()).map_err(|_| miette!("the global signature is not valid base64"))?;
    let gsig = Signature::from_bytes(&gb.as_slice().try_into().map_err(|_| miette!("the global signature has the wrong length"))?);
    let mut gm = sig.to_bytes().to_vec();
    gm.extend_from_slice(tc.as_bytes());
    vk.verify(&gm, &gsig).map_err(|_| miette!("the trusted comment does not verify"))?;
    Ok((kid_hex(&kid), tc.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_and_verify_round_trip() {
        let k = keygen(Some("t")).unwrap();
        let sk = k["secret_key"].as_str().unwrap();
        let pk = k["public_key"].as_str().unwrap();
        let s = sign(b"{\"files\":[]}", sk, "timestamp:1\tskill:x").unwrap();
        let (kid, tc) = verify(b"{\"files\":[]}", &s, pk).unwrap();
        assert_eq!(kid, k["kid"].as_str().unwrap());
        assert_eq!(tc, "timestamp:1\tskill:x");
        assert!(verify(b"{\"files\":[1]}", &s, pk).is_err());
        let other = keygen(None).unwrap();
        assert!(verify(b"{\"files\":[]}", &s, other["public_key"].as_str().unwrap()).is_err());
        assert!(verify(b"{\"files\":[]}", &s.replace("skill:x", "skill:y"), pk).is_err());
        assert_eq!(public_of(sk).unwrap(), pk);
    }
}
