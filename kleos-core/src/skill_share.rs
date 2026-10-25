/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Sharing skills (docs/specs/skills.md, "Sharing"): signatures and trusted
//! keys, signed packages, a registry other stores install from, and
//! promotion across scopes.
//!
//! **Promotion** (agent → team → workspace → org) is an audited copy, never a
//! move: the target scope gets a new version (`origin: promoted`) whose
//! provenance names the source version, the promotion and the approver.
//! Skills are instance knowledge, so a promotion waits for approval unless
//! the store's policy allows it for that target kind and the version
//! qualifies (enough uses, success rate, no scripts, clean scan); promotion
//! into the org scope always needs an explicit `approve`. Under tenancy the
//! approver is the principal's accountable human; today it is the `actor`
//! the host asserts.
//!
//! **Packages** (`kleos-skill-package/1`) carry the manifest, the files and
//! detached signatures; installing one checks every file digest, the
//! manifest digest, a signature from a trusted key (unless the policy says
//! otherwise) and runs the scanner.

use std::collections::BTreeMap;

use base64::Engine;
use miette::{bail, miette, IntoDiagnostic, Result};
use serde_json::{json, Value as JsonValue};

use crate::data::value::DataValue;
use crate::runtime::db::Db;
use crate::skill::*;
use crate::storage::Storage;

pub const PACKAGE_FORMAT: &str = "kleos-skill-package/1";
const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

fn arg<'a>(a: &'a JsonValue, k: &str) -> Option<&'a str> {
    a.get(k).and_then(|v| v.as_str()).filter(|s| !s.is_empty())
}

/// The secret key of a call: `secret_key`, else the file `key_file`, else
/// the environment variable `key_env`.
fn secret_of(args: &JsonValue) -> Result<Option<String>> {
    if let Some(k) = arg(args, "secret_key") {
        return Ok(Some(k.to_string()));
    }
    if let Some(f) = arg(args, "key_file") {
        return Ok(Some(std::fs::read_to_string(f).into_diagnostic()?.trim().to_string()));
    }
    if let Some(e) = arg(args, "key_env") {
        return Ok(Some(std::env::var(e).map_err(|_| miette!("environment variable '{e}' is not set"))?.trim().to_string()));
    }
    Ok(None)
}

impl<'s, S: Storage<'s>> Db<S> {
    // ---------------------------------------------------------- keys

    /// `trust {public_key, label?}`: verify signatures from this key.
    pub(crate) fn skill_trust(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let pk = arg(args, "public_key").ok_or_else(|| miette!("skill trust needs 'public_key' (minisign form)"))?;
        let (kid, _) = crate::skill_sign::parse_public(pk)?;
        let kid = crate::skill_sign::kid_hex(&kid);
        let line = pk.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with("untrusted comment:")).unwrap_or(pk);
        self.sk_w(
            &format!("?[kid, public, label, trusted, at] <- [[$k, $p, $l, true, $now]] :put {SKILL_KEY} {{kid => public, label, trusted, at}}"),
            vec![("k", sv(&kid)), ("p", sv(line)), ("l", sv(arg(args, "label").unwrap_or(""))), ("now", DataValue::from(now_us()))],
        )?;
        Ok(json!({"kid": kid, "trusted": true}))
    }

    pub(crate) fn skill_untrust(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let kid = arg(args, "kid").ok_or_else(|| miette!("skill untrust needs 'kid'"))?;
        self.sk_w(
            &format!("?[kid, trusted] := *{SKILL_KEY}{{kid}}, kid = $k, trusted = false :update {SKILL_KEY} {{kid => trusted}}"),
            vec![("k", sv(kid))],
        )?;
        Ok(json!({"kid": kid, "trusted": false}))
    }

    pub(crate) fn skill_keys(&'s self) -> Result<JsonValue> {
        let r = self.sk_r(&format!("?[kid, public, label, trusted, at] := *{SKILL_KEY}{{kid, public, label, trusted, at}}"), vec![])?;
        Ok(JsonValue::Array(rows_json(&r).into_iter().map(JsonValue::Object).collect()))
    }

    fn skill_key(&'s self, kid: &str) -> Result<Option<(String, bool)>> {
        let r = self.sk_r(&format!("?[public, trusted] := *{SKILL_KEY}{{kid: $k, public, trusted}}"), vec![("k", sv(kid))])?;
        Ok(r.rows.first().map(|x| (cs(&x[0]).unwrap_or_default(), x[1].get_bool().unwrap_or(false))))
    }

    /// Stored signatures of a version, each checked against its key.
    pub(crate) fn skill_sigs(&'s self, ver: &str) -> Result<JsonValue> {
        let r = self.sk_r(&format!("?[kid, signature, at] := *{SKILL_SIG}{{ver: $v, kid, signature, at}}"), vec![("v", sv(ver))])?;
        Ok(JsonValue::Array(
            r.rows.iter().map(|x| json!({"kid": jv(&x[0]), "signature": jv(&x[1]), "at": jv(&x[2])})).collect(),
        ))
    }

    /// `sign {ver | name+scope, secret_key | key_file | key_env}`: a detached
    /// signature over the version's manifest; the key's public half becomes
    /// trusted in this store.
    pub(crate) fn skill_sign(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let row = self.skill_resolve(args)?;
        let secret = secret_of(args)?.ok_or_else(|| miette!("skill sign needs 'secret_key', 'key_file' or 'key_env'"))?;
        let ver = row["ver"].as_str().unwrap_or_default().to_string();
        let entries = entries_of(&row["files"]);
        let tc = format!("timestamp:{}\tskill:{}\tdigest:{}", now_us() / 1_000_000, row["name"].as_str().unwrap_or_default(), row["digest"].as_str().unwrap_or_default());
        let sig = crate::skill_sign::sign(&manifest_bytes(&entries), &secret, &tc)?;
        let public = crate::skill_sign::public_of(&secret)?;
        let (kid, _) = crate::skill_sign::parse_public(&public)?;
        let kid = crate::skill_sign::kid_hex(&kid);
        if self.skill_key(&kid)?.is_none() {
            self.skill_trust(&json!({"public_key": public, "label": arg(args, "label").unwrap_or("signing key")}))?;
        }
        self.sk_w(
            &format!("?[ver, kid, signature, at] <- [[$v, $k, $s, $now]] :put {SKILL_SIG} {{ver, kid => signature, at}}"),
            vec![("v", sv(&ver)), ("k", sv(&kid)), ("s", sv(&sig)), ("now", DataValue::from(now_us()))],
        )?;
        Ok(json!({"ver": ver, "kid": kid, "signature": sig, "public_key": public}))
    }

    /// Check signatures over manifest bytes against the store's keys (and
    /// public keys shipped alongside, which never count as trusted by
    /// themselves).
    fn check_sigs(&'s self, manifest: &[u8], sigs: &[JsonValue]) -> Result<Vec<JsonValue>> {
        let mut out = vec![];
        for s in sigs {
            let text = s.get("signature").and_then(|v| v.as_str()).unwrap_or_default();
            let kid = s.get("kid").and_then(|v| v.as_str()).unwrap_or_default();
            let (public, trusted) = match self.skill_key(kid)? {
                Some((p, t)) => (Some(p), t),
                None => (s.get("public_key").and_then(|v| v.as_str()).map(String::from), false),
            };
            let res = match public {
                Some(p) => crate::skill_sign::verify(manifest, text, &p),
                None => Err(miette!("unknown key {kid}")),
            };
            out.push(match res {
                Ok((k, tc)) => json!({"kid": k, "valid": true, "trusted": trusted, "trusted_comment": tc}),
                Err(e) => json!({"kid": kid, "valid": false, "trusted": trusted, "error": e.to_string()}),
            });
        }
        Ok(out)
    }

    /// `verify {ver | name+scope | package}`: digests and signatures.
    pub(crate) fn skill_verify(&'s self, args: &JsonValue) -> Result<JsonValue> {
        if let Some(p) = args.get("package").filter(|p| p.is_object()) {
            let (files, entries, digest) = unpack(p)?;
            let _ = files;
            let sigs = self.check_sigs(&manifest_bytes(&entries), p.get("signatures").and_then(|s| s.as_array()).map(|a| a.as_slice()).unwrap_or(&[]))?;
            let ok = sigs.iter().any(|s| s["valid"] == true && s["trusted"] == true);
            return Ok(json!({"digest": digest, "digests_ok": true, "signatures": sigs, "verified": ok}));
        }
        let row = self.skill_resolve(args)?;
        let entries = entries_of(&row["files"]);
        let mut digests_ok = digest_of(&manifest_bytes(&entries)) == row["digest"].as_str().unwrap_or_default();
        for e in &entries {
            digests_ok &= self.skill_file(&row["files"], &e.path).is_ok();
        }
        let stored: Vec<JsonValue> = self.skill_sigs(row["ver"].as_str().unwrap_or_default())?.as_array().cloned().unwrap_or_default();
        let sigs = self.check_sigs(&manifest_bytes(&entries), &stored)?;
        let ok = digests_ok && sigs.iter().any(|s| s["valid"] == true && s["trusted"] == true);
        Ok(json!({"ver": row["ver"], "digest": row["digest"], "digests_ok": digests_ok, "signatures": sigs, "verified": ok}))
    }

    // ---------------------------------------------------------- packages

    /// `package {ver | name+scope, secret_key?|key_file?|key_env?}`: a signed,
    /// self-contained package of one version.
    pub(crate) fn skill_package(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let row = self.skill_resolve(args)?;
        if row["status"] == "quarantined" {
            bail!("skill version '{}' is quarantined and cannot be packaged", row["ver"].as_str().unwrap_or_default());
        }
        if secret_of(args)?.is_some() {
            self.skill_sign(&json!({"ver": row["ver"], "secret_key": secret_of(args)?}))?;
        }
        let ver = row["ver"].as_str().unwrap_or_default().to_string();
        let files = self.skill_files(&row["files"])?;
        let entries = entries_of(&row["files"]);
        let mut sigs = vec![];
        for s in self.skill_sigs(&ver)?.as_array().cloned().unwrap_or_default() {
            let kid = s["kid"].as_str().unwrap_or_default().to_string();
            let public = self.skill_key(&kid)?.map(|k| k.0);
            sigs.push(json!({"kid": kid, "public_key": public, "signature": s["signature"]}));
        }
        Ok(json!({
            "format": PACKAGE_FORMAT,
            "name": row["name"],
            "description": row["frontmatter"].get("description"),
            "version": row["seq"],
            "digest": row["digest"],
            "manifest": entries.iter().map(|e| json!({"path": e.path, "digest": e.digest, "size": e.size})).collect::<Vec<_>>(),
            "files": files.iter().map(|(p, b)| (p.clone(), json!(B64.encode(b)))).collect::<serde_json::Map<_, _>>(),
            "signatures": sigs,
            "provenance": {"origin": row["origin"], "scope": row["scope"], "ver": ver},
            "stats": self.skill_stats_of(&ver)?,
            "scan": row["scan"],
        }))
    }

    /// `publish {name+scope | ver, secret_key?…}`: put a package in this
    /// store's registry (served at `GET /skills/registry`).
    pub(crate) fn skill_publish(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let pkg = self.skill_package(args)?;
        let ver = pkg["provenance"]["ver"].as_str().unwrap_or_default().to_string();
        self.sk_w(
            &format!("?[name, ver, scope, digest, package, at] <- [[$n, $v, $s, $d, $p, $now]] :put {SKILL_PUB} {{name, ver => scope, digest, package, at}}"),
            vec![
                ("n", sv(pkg["name"].as_str().unwrap_or_default())),
                ("v", sv(&ver)),
                ("s", sv(pkg["provenance"]["scope"].as_str().unwrap_or_default())),
                ("d", sv(pkg["digest"].as_str().unwrap_or_default())),
                ("p", jdn(&pkg)),
                ("now", DataValue::from(now_us())),
            ],
        )?;
        Ok(json!({"name": pkg["name"], "ver": ver, "digest": pkg["digest"], "signed": pkg["signatures"].as_array().map_or(0, |a| a.len())}))
    }

    /// `registry {query?, name?, ver?}`: published packages (newest version
    /// per name); with `name`, that package in full.
    pub(crate) fn skill_registry(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let r = self.sk_r(&format!("?[name, ver, scope, digest, package, at] := *{SKILL_PUB}{{name, ver, scope, digest, package, at}}"), vec![])?;
        let mut newest: BTreeMap<String, serde_json::Map<String, JsonValue>> = BTreeMap::new();
        for o in rows_json(&r) {
            let n = o["name"].as_str().unwrap_or_default().to_string();
            if let Some(v) = arg(args, "ver") {
                if o["ver"] != v {
                    continue;
                }
            }
            if newest.get(&n).map_or(true, |x| x["at"].as_i64() < o["at"].as_i64()) {
                newest.insert(n, o);
            }
        }
        if let Some(n) = arg(args, "name") {
            return newest.remove(n).map(|o| o["package"].clone()).ok_or_else(|| miette!("no published skill '{n}'"));
        }
        let q = arg(args, "query").map(|s| s.to_lowercase());
        let list: Vec<JsonValue> = newest
            .into_values()
            .filter(|o| {
                q.as_ref().map_or(true, |q| {
                    let p = &o["package"];
                    format!("{} {}", p["name"].as_str().unwrap_or(""), p["description"].as_str().unwrap_or("")).to_lowercase().contains(q.as_str())
                })
            })
            .map(|o| {
                let p = &o["package"];
                json!({"name": o["name"], "ver": o["ver"], "digest": o["digest"], "description": p["description"], "version": p["version"],
                       "signed": p["signatures"].as_array().map_or(0, |a| a.len()), "stats": p["stats"], "published": o["at"]})
            })
            .collect();
        Ok(JsonValue::Array(list))
    }

    /// `install {package | path | url+name, scope?, require_signature?,
    /// activate?}`: verify digests, signature and scan, then store the
    /// version (`origin: installed`).
    pub(crate) fn skill_install(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let pkg: JsonValue = if let Some(p) = args.get("package").filter(|p| p.is_object()) {
            p.clone()
        } else if let Some(path) = arg(args, "path") {
            serde_json::from_slice(&std::fs::read(path).into_diagnostic()?).map_err(|e| miette!("{path} is not a skill package: {e}"))?
        } else if let Some(url) = arg(args, "url") {
            let name = arg(args, "name").ok_or_else(|| miette!("skill install from a url needs 'name'"))?;
            fetch_package(url, name, arg(args, "token"))?
        } else {
            bail!("skill install needs 'package', 'path' or 'url' + 'name'")
        };
        let (files, entries, digest) = unpack(&pkg)?;
        let sigs = self.check_sigs(&manifest_bytes(&entries), pkg.get("signatures").and_then(|s| s.as_array()).map(|a| a.as_slice()).unwrap_or(&[]))?;
        let require = args.get("require_signature").and_then(|v| v.as_bool()).or(self.skill_policy().require_signature).unwrap_or(true);
        let trusted: Vec<&JsonValue> = sigs.iter().filter(|s| s["valid"] == true && s["trusted"] == true).collect();
        if require && trusted.is_empty() {
            bail!(
                "skill package '{}' has no valid signature from a trusted key ({} signature(s) checked); `::skill trust` the publisher's key, or pass require_signature: false",
                pkg["name"].as_str().unwrap_or("?"),
                sigs.len()
            );
        }
        let scope = norm_scope(arg(args, "scope"))?;
        let actor = actor_of(args.get("actor"));
        let prov = json!({"package_digest": digest, "source": arg(args, "url").or(arg(args, "path")).unwrap_or("package"),
                          "signed_by": trusted.iter().map(|s| s["kid"].clone()).collect::<Vec<_>>(), "upstream": pkg.get("provenance")});
        let mut out = self.skill_put_files(&scope, files, "installed", None, prov, &actor, None)?;
        let ver = out["ver"].as_str().unwrap_or_default().to_string();
        for s in pkg.get("signatures").and_then(|s| s.as_array()).cloned().unwrap_or_default() {
            if let (Some(kid), Some(sig)) = (s["kid"].as_str(), s["signature"].as_str()) {
                if sigs.iter().any(|c| c["kid"] == kid && c["valid"] == true) {
                    self.sk_w(
                        &format!("?[ver, kid, signature, at] <- [[$v, $k, $s, $now]] :put {SKILL_SIG} {{ver, kid => signature, at}}"),
                        vec![("v", sv(&ver)), ("k", sv(kid)), ("s", sv(sig)), ("now", DataValue::from(now_us()))],
                    )?;
                }
            }
        }
        if args.get("activate").and_then(|v| v.as_bool()) == Some(true) && out["status"] == "candidate" {
            self.skill_activate_ver(&ver, &actor, "install")?;
            out["status"] = json!("active");
        }
        out["signatures"] = JsonValue::Array(sigs);
        Ok(out)
    }

    // ---------------------------------------------------------- promotion

    /// `promote {name+scope | ver, to, approve?, actor?, reason?}`.
    pub(crate) fn skill_promote(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let row = self.skill_resolve(args)?;
        let from = row["scope"].as_str().unwrap_or_default().to_string();
        let to = norm_scope(Some(arg(args, "to").ok_or_else(|| miette!("skill promote needs 'to' (the target scope)"))?))?;
        if scope_rank(&to) <= scope_rank(&from) {
            bail!("promotion goes to a broader scope: '{from}' ({}) → '{to}' ({}) is not broader", scope_kind(&from), scope_kind(&to));
        }
        if row["status"] == "quarantined" || row["scan"]["verdict"] == "quarantine" {
            bail!("a quarantined version cannot be promoted");
        }
        let actor = actor_of(args.get("actor"));
        let id = new_id("sp");
        let now = now_us();
        let ver = row["ver"].as_str().unwrap_or_default().to_string();
        // policy: automatic only for qualifying versions into listed kinds
        let pol = self.skill_policy();
        let stats = self.skill_outcome_counts(&[ver.clone()])?.remove(&ver).unwrap_or_default();
        let scripts = row["scan"]["scripts"].as_u64().unwrap_or(0);
        let auto = pol.auto_promote_to.iter().any(|k| k == scope_kind(&to))
            && scope_kind(&to) != "org"
            && stats.uses >= pol.auto_min_uses
            && (stats.uses == 0 || stats.successes as f64 / stats.uses as f64 >= pol.auto_min_success)
            && scripts == 0
            && row["scan"]["verdict"] == "clean";
        let by_human = args.get("approve").and_then(|v| v.as_bool()) == Some(true) && actor["human"].is_string() && scope_kind(&to) != "org";
        self.sk_w(
            &format!(
                "{{?[id, ver, scope, name, to, status, actor, approver, at, decided, reason, new_ver] <- [[$id, $v, $s, $n, $to, 'pending', $a, null, $now, null, $why, null]] \
                  :put {SKILL_PROMO} {{id => ver, scope, name, to, status, actor, approver, at, decided, reason, new_ver}}}} \
                 {{?[scope, name, at, eid, op, ver, from, actor, detail] <- [[$s, $n, $now, $eid, 'promote_request', $v, null, $a, $d]] \
                  :put {SKILL_LOG} {{scope, name, at, eid => op, ver, from, actor, detail}}}}"
            ),
            vec![
                ("id", sv(&id)),
                ("v", sv(&ver)),
                ("s", sv(&from)),
                ("n", sv(row["name"].as_str().unwrap_or_default())),
                ("to", sv(&to)),
                ("a", jdn(&actor)),
                ("now", DataValue::from(now)),
                ("why", osv(arg(args, "reason"))),
                ("eid", sv(&new_id("se"))),
                ("d", jdn(&json!({"promotion": id, "to": to}))),
            ],
        )?;
        if auto || by_human {
            let approver = if auto { json!({"policy": "auto_promote_to", "kind": scope_kind(&to)}) } else { actor.clone() };
            return self.skill_apply_promotion(&id, &approver);
        }
        Ok(json!({"promotion": id, "status": "pending", "to": to, "ver": ver,
                  "note": if scope_kind(&to) == "org" { "promotion into the org scope needs `approve` by an org admin" } else { "waiting for approval (`::skill approve {promotion, actor}`)" }}))
    }

    fn skill_promo(&'s self, id: &str) -> Result<serde_json::Map<String, JsonValue>> {
        let r = self.sk_r(
            &format!("?[id, ver, scope, name, to, status, actor, approver, at, decided, reason, new_ver] := *{SKILL_PROMO}{{id, ver, scope, name, to, status, actor, approver, at, decided, reason, new_ver}}, id = $id"),
            vec![("id", sv(id))],
        )?;
        rows_json(&r).into_iter().next().ok_or_else(|| miette!("no promotion '{id}'"))
    }

    /// The copy: a new version in the target scope with the source's exact
    /// files, then live there (if the scan allows).
    fn skill_apply_promotion(&'s self, id: &str, approver: &JsonValue) -> Result<JsonValue> {
        let p = self.skill_promo(id)?;
        if p["status"] != "pending" {
            bail!("promotion '{id}' is {}", p["status"].as_str().unwrap_or("?"));
        }
        let src = self.skill_ver(p["ver"].as_str().unwrap_or_default())?.ok_or_else(|| miette!("the promoted version is gone"))?;
        let files = self.skill_files(&src["files"])?;
        let to = p["to"].as_str().unwrap_or_default().to_string();
        let prov = json!({"promotion": id, "from_ver": src["ver"], "from_scope": src["scope"], "approver": approver, "requested_by": p["actor"]});
        let put = self.skill_put_files(&to, files, "promoted", None, prov, &p["actor"], None)?;
        let new_ver = put["ver"].as_str().unwrap_or_default().to_string();
        // signatures cover the manifest, which the copy shares
        for s in self.skill_sigs(src["ver"].as_str().unwrap_or_default())?.as_array().cloned().unwrap_or_default() {
            self.sk_w(
                &format!("?[ver, kid, signature, at] <- [[$v, $k, $s, $now]] :put {SKILL_SIG} {{ver, kid => signature, at}}"),
                vec![("v", sv(&new_ver)), ("k", sv(s["kid"].as_str().unwrap_or_default())), ("s", sv(s["signature"].as_str().unwrap_or_default())), ("now", DataValue::from(now_us()))],
            )?;
        }
        let activated = if put["status"] == "candidate" || put["status"] == "active" {
            Some(self.skill_activate_ver(&new_ver, approver, &format!("promotion {id}"))?)
        } else {
            None
        };
        let now = now_us();
        self.sk_w(
            &format!(
                "{{?[id, status, approver, decided, new_ver] <- [[$id, 'approved', $ap, $now, $nv]] :update {SKILL_PROMO} {{id => status, approver, decided, new_ver}}}} \
                 {{?[scope, name, at, eid, op, ver, from, actor, detail] <- [[$s, $n, $now, $eid, 'promoted', $v, null, $ap, $d]] \
                  :put {SKILL_LOG} {{scope, name, at, eid => op, ver, from, actor, detail}}}}"
            ),
            vec![
                ("id", sv(id)),
                ("ap", jdn(approver)),
                ("now", DataValue::from(now)),
                ("nv", sv(&new_ver)),
                ("s", sv(p["scope"].as_str().unwrap_or_default())),
                ("n", sv(p["name"].as_str().unwrap_or_default())),
                ("eid", sv(&new_id("se"))),
                ("v", sv(p["ver"].as_str().unwrap_or_default())),
                ("d", jdn(&json!({"promotion": id, "to": to, "new_ver": new_ver}))),
            ],
        )?;
        Ok(json!({"promotion": id, "status": "approved", "to": to, "new_ver": new_ver, "active": activated.is_some(), "scan": put.get("scan")}))
    }

    /// `approve {promotion, actor}`: an accountable human approves.
    pub(crate) fn skill_approve(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let id = arg(args, "promotion").or_else(|| arg(args, "id")).ok_or_else(|| miette!("skill approve needs 'promotion'"))?;
        let actor = actor_of(args.get("actor"));
        if !actor["human"].is_string() {
            bail!("skill approve needs an accountable human: pass actor: {{human: '<id>'}}");
        }
        self.skill_apply_promotion(id, &actor)
    }

    pub(crate) fn skill_reject(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let id = arg(args, "promotion").or_else(|| arg(args, "id")).ok_or_else(|| miette!("skill reject needs 'promotion'"))?;
        let p = self.skill_promo(id)?;
        if p["status"] != "pending" {
            bail!("promotion '{id}' is {}", p["status"].as_str().unwrap_or("?"));
        }
        self.sk_w(
            &format!("?[id, status, approver, decided, reason] <- [[$id, 'rejected', $a, $now, $why]] :update {SKILL_PROMO} {{id => status, approver, decided, reason}}"),
            vec![("id", sv(id)), ("a", jdn(&actor_of(args.get("actor")))), ("now", DataValue::from(now_us())), ("why", osv(arg(args, "reason")))],
        )?;
        Ok(json!({"promotion": id, "status": "rejected"}))
    }

    pub(crate) fn skill_promotions(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let r = self.sk_r(
            &format!("?[id, ver, scope, name, to, status, actor, approver, at, decided, reason, new_ver] := *{SKILL_PROMO}{{id, ver, scope, name, to, status, actor, approver, at, decided, reason, new_ver}}"),
            vec![],
        )?;
        let st = arg(args, "status");
        Ok(JsonValue::Array(rows_json(&r).into_iter().filter(|o| st.map_or(true, |s| o["status"] == s)).map(JsonValue::Object).collect()))
    }
}

/// Decode and check a package: every file against the manifest, the
/// manifest against the package digest.
fn unpack(pkg: &JsonValue) -> Result<(BTreeMap<String, Vec<u8>>, Vec<FileEntry>, String)> {
    if pkg.get("format").and_then(|f| f.as_str()) != Some(PACKAGE_FORMAT) {
        bail!("not a {PACKAGE_FORMAT} package");
    }
    let mut files = BTreeMap::new();
    for (p, b) in pkg.get("files").and_then(|f| f.as_object()).ok_or_else(|| miette!("package has no files"))? {
        check_path(p)?;
        files.insert(p.clone(), B64.decode(b.as_str().unwrap_or_default()).map_err(|_| miette!("package file '{p}' is not valid base64"))?);
    }
    let (entries, digest) = manifest(&files);
    let listed: Vec<FileEntry> = pkg
        .get("manifest")
        .and_then(|m| m.as_array())
        .ok_or_else(|| miette!("package has no manifest"))?
        .iter()
        .filter_map(|f| Some(FileEntry { path: f["path"].as_str()?.to_string(), digest: f["digest"].as_str()?.to_string(), size: f["size"].as_u64()? as usize }))
        .collect();
    let mut sorted = listed.clone();
    sorted.sort_by(|a, b| a.path.cmp(&b.path));
    if sorted != entries {
        bail!("package files do not match its manifest (a file was changed, added or removed)");
    }
    if pkg.get("digest").and_then(|d| d.as_str()) != Some(digest.as_str()) {
        bail!("package digest does not match its manifest");
    }
    Ok((files, entries, digest))
}

/// A package from another kleos server's registry.
#[cfg(feature = "requests")]
fn fetch_package(url: &str, name: &str, token: Option<&str>) -> Result<JsonValue> {
    let u = format!("{}/skills/registry/{}", url.trim_end_matches('/'), pct(name));
    let mut req = minreq::get(&u).with_timeout(60);
    if let Some(t) = token {
        req = req.with_header("x-cozo-auth", t);
    }
    let resp = req.send().map_err(|e| miette!("cannot fetch {u}: {e}"))?;
    if resp.status_code / 100 != 2 {
        bail!("{u} answered HTTP {}", resp.status_code);
    }
    let v: JsonValue = serde_json::from_str(resp.as_str().map_err(|e| miette!("{e}"))?).into_diagnostic()?;
    Ok(v.get("result").cloned().unwrap_or(v))
}

#[cfg(not(feature = "requests"))]
fn fetch_package(_url: &str, _name: &str, _token: Option<&str>) -> Result<JsonValue> {
    bail!("installing from a url needs kleos built with the `requests` feature; pass 'package' or 'path'")
}
