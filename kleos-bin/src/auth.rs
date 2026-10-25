/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Credentials of principal-mode requests (docs/specs/tenancy.md,
//! "Surfaces"):
//!
//! - RFC 9068 JWT access tokens with OAuth Actor Profile `act` chains, keys
//!   from an inline JWKS, a PEM, a JWKS URL or OIDC discovery (cached,
//!   refetched on an unknown `kid`);
//! - SPIFFE JWT-SVIDs (an issuer per trust domain);
//! - Transaction Tokens (`Txn-Token` header, draft-ietf-oauth-transaction-tokens);
//! - DPoP sender constraints (RFC 9449: `cnf.jkt` and a `DPoP` proof);
//! - per-principal API keys;
//! - an RFC 8693 token-exchange endpoint that issues kleos delegation
//!   tokens with nested `act` (actors authenticate with `private_key_jwt`,
//!   a JWT-SVID, an access token or an API key);
//! - RFC 9728 protected-resource metadata.
//!
//! kleos is a resource server: it validates the audience and never forwards
//! an inbound token.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine;
use jsonwebtoken::jwk::{AlgorithmParameters, Jwk, JwkSet};
use jsonwebtoken::{decode, decode_header, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use kleos::tenancy::{ActorLink, Assurance, PrincipalContext, TokenGrant, MAX_CHAIN};
use kleos::DbInstance;
use serde_derive::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// `--tenancy <config.json>`.
#[derive(Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct TenancyConfig {
    /// This server as an OAuth resource (RFC 8707): tokens must name it in
    /// `aud`; RFC 9728 metadata publishes it.
    pub resource: String,
    #[serde(default)]
    pub authorization_servers: Vec<String>,
    #[serde(default)]
    pub issuers: Vec<IssuerConfig>,
    /// Accept `kl_…` API keys (`::tenant apikey_create`).
    #[serde(default = "yes")]
    pub api_keys: bool,
    /// Workspace of requests that name none (`X-Kleos-Workspace` or the
    /// token's `kleos_ws` claim).
    #[serde(default)]
    pub default_workspace: Option<String>,
    /// The issuer id of the tokens kleos mints (token exchange); default
    /// `resource`.
    #[serde(default)]
    pub issuer: Option<String>,
    /// The environment variable holding the HS256 key of kleos-minted
    /// tokens. Without it, token exchange is off.
    #[serde(default)]
    pub signing_secret_env: Option<String>,
    /// Lifetime of minted tokens (seconds; default 600).
    #[serde(default)]
    pub token_ttl_secs: Option<u64>,
    /// The audience Transaction Tokens must carry (the trust domain);
    /// default `resource`.
    #[serde(default)]
    pub txn_audience: Option<String>,
}

fn yes() -> bool {
    true
}

/// A trusted token issuer.
#[derive(Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct IssuerConfig {
    pub iss: String,
    /// An inline JWKS (`{"keys": [...]}`).
    #[serde(default)]
    pub jwks: Option<Value>,
    /// A JWKS URL, fetched and cached (`jwks_ttl_secs`, default 300) and
    /// refetched when a token names an unknown `kid` (at most every 10 s).
    #[serde(default)]
    pub jwks_uri: Option<String>,
    /// Find `jwks_uri` through OIDC discovery (`<iss>/.well-known/openid-configuration`).
    #[serde(default)]
    pub discovery: bool,
    #[serde(default)]
    pub jwks_ttl_secs: Option<u64>,
    /// A public key in PEM (RSA or EC, per `alg`).
    #[serde(default)]
    pub pem: Option<String>,
    /// `RS256` (default), `ES256`, `EdDSA` or `HS256`.
    #[serde(default)]
    pub alg: Option<String>,
    /// HS256 only: the environment variable holding the shared secret.
    #[serde(default)]
    pub secret_env: Option<String>,
    /// How `sub` maps to a principal: `ident` (`kleos_tn_ident`, the
    /// default) or `direct` (`sub` is the principal id).
    #[serde(default)]
    pub map: Option<String>,
    /// A SPIFFE trust domain: JWT-SVIDs whose `sub` is
    /// `spiffe://<domain>/…` are checked against this issuer's keys (they
    /// may carry no `iss`).
    #[serde(default)]
    pub spiffe_trust_domain: Option<String>,
    /// This issuer is a Transaction Token service.
    #[serde(default)]
    pub txn: bool,
}

/// A refused credential: HTTP status, OAuth error code, description.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthError {
    pub status: u16,
    pub error: &'static str,
    pub description: String,
}

fn unauthorized(d: impl Into<String>) -> AuthError {
    AuthError { status: 401, error: "invalid_token", description: d.into() }
}

fn bad_request(error: &'static str, d: impl Into<String>) -> AuthError {
    AuthError { status: 400, error, description: d.into() }
}

/// The request a credential arrives with (for DPoP proofs).
#[derive(Debug, Clone, Default)]
pub struct RequestInfo {
    pub method: String,
    /// The full request URL without query and fragment.
    pub url: String,
}

impl TenancyConfig {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("bad tenancy config {path}: {e}"))
    }

    /// The issuer id of kleos-minted tokens.
    pub fn self_issuer(&self) -> String {
        self.issuer.clone().unwrap_or_else(|| self.resource.clone())
    }

    fn signing_secret(&self) -> Option<Vec<u8>> {
        self.signing_secret_env.as_ref().and_then(|e| std::env::var(e).ok()).filter(|s| s.len() >= 32).map(|s| s.into_bytes())
    }

    /// RFC 9728 protected-resource metadata.
    pub fn metadata(&self) -> Value {
        let mut servers = self.authorization_servers.clone();
        if self.signing_secret_env.is_some() && !servers.contains(&self.self_issuer()) {
            servers.push(self.self_issuer());
        }
        json!({
            "resource": self.resource,
            "authorization_servers": servers,
            "bearer_methods_supported": ["header"],
            "scopes_supported": ["kleos:read", "kleos:write", "kleos:promote", "kleos:admin"],
            "dpop_signing_alg_values_supported": ["ES256", "RS256", "EdDSA"],
            "resource_name": "kleos",
            "resource_documentation": "https://github.com/kleos-db/kleos/blob/main/docs/specs/tenancy.md",
        })
    }

    /// RFC 8414 metadata of kleos's own token endpoint (token exchange).
    pub fn as_metadata(&self, origin: &str) -> Value {
        json!({
            "issuer": self.self_issuer(),
            "token_endpoint": format!("{}/oauth/token", origin.trim_end_matches('/')),
            "grant_types_supported": ["urn:ietf:params:oauth:grant-type:token-exchange"],
            "token_endpoint_auth_methods_supported": ["private_key_jwt", "none"],
            "token_endpoint_auth_signing_alg_values_supported": ["ES256", "RS256", "EdDSA"],
            "dpop_signing_alg_values_supported": ["ES256", "RS256", "EdDSA"],
        })
    }

    /// The `WWW-Authenticate` challenge of a 401/403 (RFC 9728 §5.1).
    pub fn challenge(&self, origin: &str, err: Option<&AuthError>) -> String {
        let mut c = format!("Bearer resource_metadata=\"{}/.well-known/oauth-protected-resource\"", origin.trim_end_matches('/'));
        if let Some(e) = err {
            c.push_str(&format!(", error=\"{}\", error_description=\"{}\"", e.error, e.description.replace('"', "'")));
        }
        c
    }

    fn issuer(&self, iss: &str) -> Option<&IssuerConfig> {
        self.issuers.iter().find(|i| i.iss == iss)
    }
}

fn link_kind(profile: Option<&str>, registered: Option<&str>) -> String {
    // `sub_profile` is space-delimited; an agent anywhere in it is an agent
    // (the more restrictive reading)
    let p = profile.unwrap_or_default();
    if p.split(' ').any(|x| x == "ai_agent") {
        "ai_agent".into()
    } else if p.split(' ').any(|x| x == "service") {
        "service".into()
    } else if p.split(' ').any(|x| x == "user") {
        "human".into()
    } else {
        match registered {
            Some("agent") => "ai_agent".into(),
            Some("service") => "service".into(),
            _ => "human".into(),
        }
    }
}

fn profile_of(kind: &str) -> &'static str {
    match kind {
        "agent" | "ai_agent" => "ai_agent",
        "service" => "service",
        _ => "user",
    }
}

/// How a verified token's subjects map to principals.
#[derive(Clone, Copy)]
enum Map<'a> {
    Issuer(&'a IssuerConfig),
    /// kleos's own tokens: subjects are principal ids.
    Direct,
}

/// The kleos principal behind `(iss, sub)`.
fn map_principal(db: &DbInstance, map: Map<'_>, iss: &str, sub: &str) -> Option<String> {
    if let Ok(Some(p)) = db.resolve_ident(iss, sub) {
        return Some(p);
    }
    let direct = match map {
        Map::Direct => true,
        Map::Issuer(i) => i.map.as_deref() == Some("direct"),
    };
    (direct && db.principal_kind(sub).ok().flatten().is_some()).then(|| sub.to_string())
}

fn grant_of(scope: Option<&str>) -> Option<TokenGrant> {
    let words: Vec<&str> = scope.unwrap_or_default().split(' ').filter(|w| w.starts_with("kleos:")).collect();
    if words.is_empty() {
        return None;
    }
    let mut actions = vec![];
    let mut kinds = vec![];
    for w in words {
        match w.strip_prefix("kleos:scope:") {
            Some(k) => kinds.push(k.to_string()),
            None => actions.push(w.trim_start_matches("kleos:").to_string()),
        }
    }
    Some(TokenGrant { actions: (!actions.is_empty()).then_some(actions), kinds: (!kinds.is_empty()).then_some(kinds) })
}

/// The intersection of two grants (a credential and a Transaction Token).
fn meet(a: Option<TokenGrant>, b: Option<TokenGrant>) -> Option<TokenGrant> {
    let both = |x: Option<Vec<String>>, y: Option<Vec<String>>| match (x, y) {
        (Some(x), Some(y)) => Some(x.into_iter().filter(|v| y.contains(v)).collect()),
        (x, None) => x,
        (None, y) => y,
    };
    match (a, b) {
        (Some(a), Some(b)) => Some(TokenGrant { actions: both(a.actions, b.actions), kinds: both(a.kinds, b.kinds) }),
        (a, None) => a,
        (None, b) => b,
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn b64url(b: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
}

fn payload_of(token: &str) -> Option<Value> {
    let p = token.split('.').nth(1)?;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(p.trim_end_matches('=')).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

// ------------------------------------------------------------------ JWKS

struct CachedJwks {
    set: JwkSet,
    fetched: Instant,
    last_miss: Option<Instant>,
}

static JWKS_CACHE: Mutex<Option<HashMap<String, CachedJwks>>> = Mutex::new(None);

fn http_get_json(url: &str) -> Result<Value, AuthError> {
    let resp = minreq::get(url).with_timeout(10).send().map_err(|e| unauthorized(format!("cannot fetch {url}: {e}")))?;
    if resp.status_code != 200 {
        return Err(unauthorized(format!("{url} answered HTTP {}", resp.status_code)));
    }
    serde_json::from_str(resp.as_str().map_err(|e| unauthorized(e.to_string()))?).map_err(|e| unauthorized(format!("{url}: {e}")))
}

/// The key `kid` of `issuer`'s JWKS URL: from the cache while fresh, else
/// fetched; an unknown `kid` triggers a refetch (key rotation), rate
/// limited.
fn jwks_key(issuer: &IssuerConfig, kid: Option<&str>) -> Result<Jwk, AuthError> {
    let uri = match &issuer.jwks_uri {
        Some(u) => u.clone(),
        None => {
            let d = http_get_json(&format!("{}/.well-known/openid-configuration", issuer.iss.trim_end_matches('/')))?;
            d.get("jwks_uri").and_then(|v| v.as_str()).ok_or_else(|| unauthorized("discovery document has no jwks_uri"))?.to_string()
        }
    };
    let ttl = Duration::from_secs(issuer.jwks_ttl_secs.unwrap_or(300));
    let find = |set: &JwkSet| -> Option<Jwk> {
        match kid {
            Some(k) => set.find(k).cloned(),
            None => set.keys.first().cloned(),
        }
    };
    {
        let mut g = JWKS_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        let map = g.get_or_insert_with(HashMap::new);
        if let Some(c) = map.get_mut(&uri) {
            let fresh = c.fetched.elapsed() < ttl;
            if fresh {
                if let Some(k) = find(&c.set) {
                    return Ok(k);
                }
                // an unknown kid: refetch, but not more than every 10 s
                if c.last_miss.is_some_and(|t| t.elapsed() < Duration::from_secs(10)) {
                    return Err(unauthorized("no matching key"));
                }
                c.last_miss = Some(Instant::now());
            }
        }
    }
    let set: JwkSet = serde_json::from_value(http_get_json(&uri)?).map_err(|e| unauthorized(format!("bad JWKS at {uri}: {e}")))?;
    let key = find(&set);
    let mut g = JWKS_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let map = g.get_or_insert_with(HashMap::new);
    let last_miss = map.get(&uri).and_then(|c| c.last_miss);
    map.insert(uri, CachedJwks { set, fetched: Instant::now(), last_miss });
    key.ok_or_else(|| unauthorized("no matching key"))
}

fn alg_of(issuer: &IssuerConfig) -> Result<Algorithm, AuthError> {
    Ok(match issuer.alg.as_deref().unwrap_or("RS256") {
        "RS256" => Algorithm::RS256,
        "ES256" => Algorithm::ES256,
        "EdDSA" => Algorithm::EdDSA,
        "HS256" => Algorithm::HS256,
        other => return Err(unauthorized(format!("unsupported alg '{other}' configured"))),
    })
}

fn key_of(issuer: &IssuerConfig, alg: Algorithm, kid: Option<&str>) -> Result<DecodingKey, AuthError> {
    if let Some(env) = &issuer.secret_env {
        let secret = std::env::var(env).map_err(|_| unauthorized("issuer secret is not set"))?;
        return Ok(DecodingKey::from_secret(secret.as_bytes()));
    }
    if let Some(jwks) = &issuer.jwks {
        let set: JwkSet = serde_json::from_value(jwks.clone()).map_err(|e| unauthorized(format!("bad JWKS: {e}")))?;
        let jwk = match kid {
            Some(k) => set.find(k),
            None => set.keys.first(),
        }
        .ok_or_else(|| unauthorized("no matching key"))?;
        return DecodingKey::from_jwk(jwk).map_err(|e| unauthorized(format!("bad key: {e}")));
    }
    if issuer.jwks_uri.is_some() || issuer.discovery {
        return DecodingKey::from_jwk(&jwks_key(issuer, kid)?).map_err(|e| unauthorized(format!("bad key: {e}")));
    }
    if let Some(pem) = &issuer.pem {
        return match alg {
            Algorithm::ES256 => DecodingKey::from_ec_pem(pem.as_bytes()),
            Algorithm::EdDSA => DecodingKey::from_ed_pem(pem.as_bytes()),
            _ => DecodingKey::from_rsa_pem(pem.as_bytes()),
        }
        .map_err(|e| unauthorized(format!("bad key: {e}")));
    }
    Err(unauthorized("issuer has no key configured"))
}

/// A verified token: its claims, its issuer id, how its subjects map.
struct Verified<'a> {
    claims: Value,
    iss: String,
    map: Map<'a>,
}

/// Verify a JWT for `aud`: a kleos-minted token, or one of a configured
/// issuer (a JWT-SVID is found by its SPIFFE trust domain).
fn verify<'a>(cfg: &'a TenancyConfig, token: &str, aud: &str, txn: bool) -> Result<Verified<'a>, AuthError> {
    if token.split('.').count() != 3 {
        return Err(unauthorized("not a JWT or an API key"));
    }
    let unverified = payload_of(token).ok_or_else(|| unauthorized("malformed token"))?;
    let head = decode_header(token).map_err(|e| unauthorized(format!("bad token header: {e}")))?;
    let claimed_iss = unverified.get("iss").and_then(|v| v.as_str()).map(String::from);
    let mut v = |alg: Algorithm, key: DecodingKey, iss: Option<&str>| -> Result<Value, AuthError> {
        if head.alg != alg {
            return Err(unauthorized("unexpected token algorithm"));
        }
        let mut val = Validation::new(alg);
        val.set_audience(&[aud]);
        if let Some(i) = iss {
            val.set_issuer(&[i]);
        }
        val.required_spec_claims = ["exp", "aud"].iter().map(|x| x.to_string()).collect();
        val.leeway = 30;
        Ok(decode::<Value>(token, &key, &val).map_err(|e| unauthorized(format!("token rejected: {e}")))?.claims)
    };
    // kleos's own delegation tokens
    if !txn && claimed_iss.as_deref() == Some(cfg.self_issuer().as_str()) {
        let secret = cfg.signing_secret().ok_or_else(|| unauthorized("token exchange is not configured"))?;
        let claims = v(Algorithm::HS256, DecodingKey::from_secret(&secret), Some(&cfg.self_issuer()))?;
        return Ok(Verified { claims, iss: cfg.self_issuer(), map: Map::Direct });
    }
    let issuer = match &claimed_iss {
        Some(i) => cfg.issuer(i),
        None => {
            // a JWT-SVID may have no `iss`: its trust domain names the issuer
            let sub = unverified.get("sub").and_then(|v| v.as_str()).unwrap_or_default();
            let td = sub.strip_prefix("spiffe://").and_then(|r| r.split('/').next()).unwrap_or_default();
            cfg.issuers.iter().find(|i| i.spiffe_trust_domain.as_deref() == Some(td) && !td.is_empty())
        }
    }
    .filter(|i| i.txn == txn)
    .ok_or_else(|| unauthorized(format!("untrusted issuer '{}'", claimed_iss.clone().unwrap_or_default())))?;
    let alg = alg_of(issuer)?;
    let key = key_of(issuer, alg, head.kid.as_deref())?;
    let claims = v(alg, key, claimed_iss.as_deref())?;
    if let Some(td) = &issuer.spiffe_trust_domain {
        let sub = claims.get("sub").and_then(|v| v.as_str()).unwrap_or_default();
        if !sub.starts_with(&format!("spiffe://{td}/")) {
            return Err(unauthorized("the SVID's subject is outside its trust domain"));
        }
    }
    let iss = claimed_iss.unwrap_or_else(|| format!("spiffe://{}", issuer.spiffe_trust_domain.clone().unwrap_or_default()));
    Ok(Verified { claims, iss, map: Map::Issuer(issuer) })
}

// ------------------------------------------------------------------ DPoP

static DPOP_JTI: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// The RFC 7638 thumbprint of a public JWK.
pub fn jwk_thumbprint(jwk: &Jwk) -> Option<String> {
    let canon = match &jwk.algorithm {
        AlgorithmParameters::EllipticCurve(ec) => {
            let crv = serde_json::to_value(&ec.curve).ok()?;
            format!(r#"{{"crv":{},"kty":"EC","x":"{}","y":"{}"}}"#, crv, ec.x, ec.y)
        }
        AlgorithmParameters::RSA(r) => format!(r#"{{"e":"{}","kty":"RSA","n":"{}"}}"#, r.e, r.n),
        AlgorithmParameters::OctetKeyPair(o) => {
            let crv = serde_json::to_value(&o.curve).ok()?;
            format!(r#"{{"crv":{},"kty":"OKP","x":"{}"}}"#, crv, o.x)
        }
        AlgorithmParameters::OctetKey(_) => return None,
    };
    Some(b64url(&Sha256::digest(canon.as_bytes())))
}

/// Check a DPoP proof (RFC 9449) for `req`; `access_token` binds `ath`.
/// Returns the proof key's thumbprint.
pub fn verify_dpop(proof: &str, req: &RequestInfo, access_token: Option<&str>) -> Result<String, AuthError> {
    let bad = |d: &str| AuthError { status: 401, error: "invalid_dpop_proof", description: d.to_string() };
    let head = decode_header(proof).map_err(|_| bad("malformed DPoP proof"))?;
    if head.typ.as_deref() != Some("dpop+jwt") {
        return Err(bad("DPoP proof typ must be dpop+jwt"));
    }
    let jwk = head.jwk.clone().ok_or_else(|| bad("DPoP proof has no jwk"))?;
    if matches!(jwk.algorithm, AlgorithmParameters::OctetKey(_)) || head.alg == Algorithm::HS256 {
        return Err(bad("DPoP proofs use an asymmetric key"));
    }
    let key = DecodingKey::from_jwk(&jwk).map_err(|_| bad("bad DPoP key"))?;
    let mut val = Validation::new(head.alg);
    val.required_spec_claims = Default::default();
    val.validate_exp = false;
    val.validate_aud = false;
    let c = decode::<Value>(proof, &key, &val).map_err(|e| bad(&format!("DPoP proof rejected: {e}")))?.claims;
    if c.get("htm").and_then(|v| v.as_str()) != Some(req.method.as_str()) {
        return Err(bad("DPoP htm does not match the request"));
    }
    if c.get("htu").and_then(|v| v.as_str()).map(|u| u.split(['?', '#']).next().unwrap_or(u).trim_end_matches('/')) != Some(req.url.trim_end_matches('/')) {
        return Err(bad("DPoP htu does not match the request"));
    }
    let iat = c.get("iat").and_then(|v| v.as_u64()).ok_or_else(|| bad("DPoP proof has no iat"))?;
    let now = now_secs();
    if iat + 300 < now || iat > now + 60 {
        return Err(bad("DPoP proof is stale"));
    }
    if let Some(at) = access_token {
        if c.get("ath").and_then(|v| v.as_str()) != Some(b64url(&Sha256::digest(at.as_bytes())).as_str()) {
            return Err(bad("DPoP ath does not match the access token"));
        }
    }
    let jti = c.get("jti").and_then(|v| v.as_str()).ok_or_else(|| bad("DPoP proof has no jti"))?.to_string();
    let mut g = DPOP_JTI.lock().unwrap_or_else(|e| e.into_inner());
    let seen = g.get_or_insert_with(HashMap::new);
    seen.retain(|_, t| *t + 600 > now);
    if seen.insert(jti, now).is_some() {
        return Err(bad("DPoP proof replayed"));
    }
    jwk_thumbprint(&jwk).ok_or_else(|| bad("bad DPoP key"))
}

// ------------------------------------------------------------------ authentication

/// The chain of a verified token, innermost (the subject) first.
fn chain_of(db: &DbInstance, cfg: &TenancyConfig, v: &Verified<'_>) -> Result<Vec<ActorLink>, AuthError> {
    let claims = &v.claims;
    let sub = claims.get("sub").and_then(|x| x.as_str()).ok_or_else(|| unauthorized("token has no sub"))?.to_string();
    let mut acts = vec![];
    let mut cur = claims.get("act");
    while let Some(a) = cur {
        if acts.len() >= MAX_CHAIN {
            return Err(bad_request("invalid_request", format!("actor chain deeper than {MAX_CHAIN}")));
        }
        acts.push(a.clone());
        cur = a.get("act");
    }
    let sp = map_principal(db, v.map, &v.iss, &sub);
    let sk = sp.as_deref().and_then(|p| db.principal_kind(p).ok().flatten());
    let mut chain = vec![ActorLink {
        kind: link_kind(claims.get("sub_profile").and_then(|x| x.as_str()), sk.as_deref()),
        iss: Some(v.iss.clone()),
        sub,
        principal: sp,
        assurance: Assurance::Verified,
    }];
    // the nested `act` objects run from the presenter (outermost) inwards
    for a in acts.iter().rev() {
        let asub = a.get("sub").and_then(|x| x.as_str()).ok_or_else(|| bad_request("invalid_request", "an act object needs sub"))?;
        let aiss = a.get("iss").and_then(|x| x.as_str()).unwrap_or(&v.iss).to_string();
        let map = if aiss == cfg.self_issuer() { Map::Direct } else { cfg.issuer(&aiss).map(Map::Issuer).unwrap_or(v.map) };
        let p = map_principal(db, map, &aiss, asub);
        let pk = p.as_deref().and_then(|x| db.principal_kind(x).ok().flatten());
        chain.push(ActorLink {
            kind: link_kind(a.get("sub_profile").and_then(|x| x.as_str()), pk.as_deref()),
            iss: Some(aiss),
            sub: asub.to_string(),
            principal: p,
            assurance: Assurance::IssuerEndorsed,
        });
    }
    // an AI agent presenting the token must be a known principal: its
    // permissions are part of the intersection
    if chain.len() > 1 && chain.last().is_some_and(|l| l.kind == "ai_agent" && l.principal.is_none()) {
        return Err(AuthError { status: 403, error: "actor_unauthorized", description: "the presenting agent is not a kleos principal".into() });
    }
    Ok(chain)
}

/// A context from a chain: the subject's principal by kind; the acting
/// agent is the outermost agent link that maps to a principal.
fn ctx_of(chain: Vec<ActorLink>) -> Result<PrincipalContext, AuthError> {
    let Some(sp) = chain[0].principal.clone() else {
        return Err(AuthError { status: 403, error: "insufficient_scope", description: format!("subject '{}' is not a kleos principal", chain[0].sub) });
    };
    let mut ctx = PrincipalContext::default();
    match chain[0].kind.as_str() {
        "ai_agent" => ctx.agent = Some(sp),
        "service" => ctx.service = Some(sp),
        _ => ctx.human = Some(sp),
    }
    if let Some(a) = chain.iter().skip(1).rev().find(|l| l.kind == "ai_agent" && l.principal.is_some()) {
        ctx.agent = a.principal.clone();
    }
    ctx.chain = chain;
    Ok(ctx)
}

fn apikey_ctx(cfg: &TenancyConfig, db: &DbInstance, key: &str, header: &dyn Fn(&str) -> Option<String>) -> Result<PrincipalContext, AuthError> {
    if !cfg.api_keys {
        return Err(unauthorized("API keys are disabled"));
    }
    let (p, actions) = db.resolve_apikey(key).ok().flatten().ok_or_else(|| unauthorized("unknown or expired API key"))?;
    let kind = db.principal_kind(&p).ok().flatten().ok_or_else(|| unauthorized("the key's principal is not active"))?;
    let mut ctx = PrincipalContext::default();
    let mut chain = vec![];
    // the key holder is authenticated; whom it acts for is only claimed
    if let (Some(h), "agent") = (header("x-kleos-on-behalf-of"), kind.as_str()) {
        chain.push(ActorLink { kind: "human".into(), iss: None, sub: h.clone(), principal: Some(h.clone()), assurance: Assurance::SelfAsserted });
        ctx.human = Some(h);
    }
    chain.push(ActorLink { kind: link_kind(None, Some(&kind)), iss: None, sub: p.clone(), principal: Some(p.clone()), assurance: Assurance::Verified });
    match kind.as_str() {
        "agent" => ctx.agent = Some(p),
        "service" => ctx.service = Some(p),
        _ => ctx.human = Some(p),
    }
    ctx.chain = chain;
    ctx.token = (!actions.is_empty()).then(|| TokenGrant { actions: Some(actions), kinds: None });
    Ok(ctx)
}

/// Authenticate a bearer credential into a principal context (no DPoP
/// request information: sender-constrained tokens are refused).
pub fn authenticate(cfg: &TenancyConfig, db: &DbInstance, bearer: &str, header: &dyn Fn(&str) -> Option<String>) -> Result<PrincipalContext, AuthError> {
    authenticate_request(cfg, db, Some(bearer), header, &RequestInfo::default())
}

/// Authenticate a request: its `Authorization` credential (`bearer`: an
/// API key, a JWT access token, a JWT-SVID or a kleos delegation token)
/// and/or its `Txn-Token`. `header` reads request headers
/// (`X-Kleos-Workspace`, `X-Kleos-Run`, `X-Kleos-On-Behalf-Of`, `DPoP`,
/// `Txn-Token`).
pub fn authenticate_request(
    cfg: &TenancyConfig,
    db: &DbInstance,
    bearer: Option<&str>,
    header: &dyn Fn(&str) -> Option<String>,
    req: &RequestInfo,
) -> Result<PrincipalContext, AuthError> {
    let txn_token = header("txn-token");
    let mut ctx = match bearer {
        Some(b) if b.starts_with("kl_") => apikey_ctx(cfg, db, b, header)?,
        Some(b) => {
            let v = verify(cfg, b, &cfg.resource, false)?;
            // a sender-constrained token needs its DPoP proof
            if let Some(jkt) = v.claims.pointer("/cnf/jkt").and_then(|x| x.as_str()) {
                let proof = header("dpop").ok_or_else(|| AuthError { status: 401, error: "invalid_dpop_proof", description: "this token needs a DPoP proof".into() })?;
                if verify_dpop(&proof, req, Some(b))? != jkt {
                    return Err(AuthError { status: 401, error: "invalid_dpop_proof", description: "the DPoP key is not the token's".into() });
                }
            }
            let mut c = ctx_of(chain_of(db, cfg, &v)?)?;
            c.workspace = v.claims.get("kleos_ws").and_then(|x| x.as_str()).map(String::from);
            c.run = v.claims.get("kleos_run").and_then(|x| x.as_str()).map(String::from);
            c.token = grant_of(v.claims.get("scope").and_then(|x| x.as_str()));
            c.client_id = v.claims.get("client_id").and_then(|x| x.as_str()).map(String::from);
            c.txn = v.claims.get("txn").and_then(|x| x.as_str()).map(String::from);
            c
        }
        None => PrincipalContext::default(),
    };
    // a Transaction Token: its subject must be the credential's, its scopes
    // narrow it, its `txn` correlates the audit, its requesting workload
    // joins the chain
    if let Some(t) = txn_token {
        let aud = cfg.txn_audience.clone().unwrap_or_else(|| cfg.resource.clone());
        let v = verify(cfg, &t, &aud, true)?;
        let tchain = chain_of(db, cfg, &v)?;
        let tsub = tchain[0].principal.clone();
        if bearer.is_some() {
            let csub = ctx.chain.first().and_then(|l| l.principal.clone());
            if csub != tsub {
                return Err(unauthorized("the Transaction Token's subject is not the credential's"));
            }
            ctx.token = meet(ctx.token.take(), grant_of(v.claims.get("scope").and_then(|x| x.as_str())));
        } else {
            let ws = ctx.workspace.take();
            ctx = ctx_of(tchain)?;
            ctx.workspace = ws;
            ctx.token = grant_of(v.claims.get("scope").and_then(|x| x.as_str()));
        }
        ctx.txn = v.claims.get("txn").and_then(|x| x.as_str()).map(String::from);
        if let Some(wl) = v.claims.get("req_wl").and_then(|x| x.as_str()) {
            let p = map_principal(db, v.map, &v.iss, wl);
            ctx.chain.push(ActorLink { kind: "service".into(), iss: Some(v.iss.clone()), sub: wl.to_string(), principal: p, assurance: Assurance::IssuerEndorsed });
        }
    } else if bearer.is_none() {
        return Err(unauthorized("no credential"));
    }
    if let Some(w) = header("x-kleos-workspace").or_else(|| ctx.workspace.clone()).or_else(|| cfg.default_workspace.clone()) {
        ctx.workspace = Some(w);
    }
    if let Some(r) = header("x-kleos-run") {
        ctx.run = Some(r);
    }
    Ok(ctx)
}

/// Whether a context's credential may write.
pub fn may_write(ctx: &PrincipalContext) -> bool {
    ctx.token.as_ref().and_then(|t| t.actions.as_ref()).map_or(true, |a| a.iter().any(|x| x == "write" || x == "promote" || x == "admin"))
}

// ------------------------------------------------------------------ token exchange

/// The actor of a token request: `client_assertion` (`private_key_jwt`:
/// signed with the principal's registered key, `iss = sub = <principal>`,
/// `aud` = the token endpoint or the resource), or `actor_token` (a JWT-SVID,
/// an access token or a kleos token of an agent, or an API key).
fn actor_of(cfg: &TenancyConfig, db: &DbInstance, form: &HashMap<String, String>, req: &RequestInfo) -> Result<Option<(String, String)>, AuthError> {
    if let Some(a) = form.get("client_assertion") {
        if form.get("client_assertion_type").map(|s| s.as_str()) != Some("urn:ietf:params:oauth:client-assertion-type:jwt-bearer") {
            return Err(bad_request("invalid_client", "unsupported client_assertion_type"));
        }
        let claims = payload_of(a).ok_or_else(|| bad_request("invalid_client", "malformed client assertion"))?;
        let id = claims.get("sub").and_then(|x| x.as_str()).ok_or_else(|| bad_request("invalid_client", "client assertion has no sub"))?.to_string();
        if claims.get("iss").and_then(|x| x.as_str()) != Some(id.as_str()) {
            return Err(bad_request("invalid_client", "client assertion iss must equal sub"));
        }
        let jwks = db.principal_jwks(&id).ok().flatten().ok_or_else(|| bad_request("invalid_client", format!("'{id}' has no registered keys")))?;
        let set: JwkSet = serde_json::from_value(jwks).map_err(|_| bad_request("invalid_client", "bad registered keys"))?;
        let head = decode_header(a).map_err(|_| bad_request("invalid_client", "malformed client assertion"))?;
        let jwk = match &head.kid {
            Some(k) => set.find(k),
            None => set.keys.first(),
        }
        .ok_or_else(|| bad_request("invalid_client", "no matching registered key"))?;
        if matches!(jwk.algorithm, AlgorithmParameters::OctetKey(_)) {
            return Err(bad_request("invalid_client", "private_key_jwt needs an asymmetric key"));
        }
        let key = DecodingKey::from_jwk(jwk).map_err(|_| bad_request("invalid_client", "bad registered key"))?;
        let mut val = Validation::new(head.alg);
        let auds: Vec<&str> = [req.url.as_str(), cfg.resource.as_str()].into_iter().filter(|x| !x.is_empty()).collect();
        val.set_audience(&auds);
        val.required_spec_claims = ["exp", "aud"].iter().map(|x| x.to_string()).collect();
        decode::<Value>(a, &key, &val).map_err(|e| bad_request("invalid_client", format!("client assertion rejected: {e}")))?;
        let kind = db.principal_kind(&id).ok().flatten().ok_or_else(|| bad_request("invalid_client", "inactive principal"))?;
        return Ok(Some((id, kind)));
    }
    if let Some(t) = form.get("actor_token") {
        let ctx = if t.starts_with("kl_") {
            apikey_ctx(cfg, db, t, &|_| None)?
        } else {
            ctx_of(chain_of(db, cfg, &verify(cfg, t, &cfg.resource, false)?)?)?
        };
        let id = ctx.agent.or(ctx.service).or(ctx.human).ok_or_else(|| bad_request("invalid_request", "the actor token names no principal"))?;
        let kind = db.principal_kind(&id).ok().flatten().unwrap_or_default();
        return Ok(Some((id, kind)));
    }
    Ok(None)
}

/// RFC 8693 token exchange: a kleos delegation token whose `sub` is the
/// subject token's principal and whose `act` adds the authenticated actor
/// outside the subject token's own chain (inner links are kept, never
/// dropped). Scopes may only narrow; a DPoP proof on the request binds the
/// token to that key. `form` is the request's form fields.
pub fn token_exchange(
    cfg: &TenancyConfig,
    db: &DbInstance,
    form: &HashMap<String, String>,
    header: &dyn Fn(&str) -> Option<String>,
    req: &RequestInfo,
) -> Result<Value, AuthError> {
    if form.get("grant_type").map(|s| s.as_str()) != Some("urn:ietf:params:oauth:grant-type:token-exchange") {
        return Err(bad_request("unsupported_grant_type", "only token exchange is supported"));
    }
    let secret = cfg.signing_secret().ok_or_else(|| bad_request("invalid_request", "token exchange is not configured (signing_secret_env, 32+ bytes)"))?;
    for k in ["audience", "resource"] {
        if let Some(a) = form.get(k) {
            if a != &cfg.resource {
                return Err(bad_request("invalid_target", "kleos issues tokens for itself only"));
            }
        }
    }
    let st = form.get("subject_token").ok_or_else(|| bad_request("invalid_request", "subject_token is required"))?;
    let sv = verify(cfg, st, &cfg.resource, false).map_err(|e| bad_request("invalid_grant", e.description))?;
    let schain = chain_of(db, cfg, &sv)?;
    let subject = schain[0].principal.clone().ok_or_else(|| bad_request("invalid_grant", "the subject is not a kleos principal"))?;
    let skind = db.principal_kind(&subject).ok().flatten().unwrap_or_default();
    let actor = actor_of(cfg, db, form, req)?;
    // scopes only narrow
    let have: Option<Vec<String>> = sv.claims.get("scope").and_then(|x| x.as_str()).map(|s| s.split(' ').filter(|w| !w.is_empty()).map(String::from).collect());
    let want: Option<Vec<String>> = form.get("scope").map(|s| s.split(' ').filter(|w| !w.is_empty()).map(String::from).collect());
    let scope = match (have, want) {
        (Some(h), Some(w)) => {
            if let Some(extra) = w.iter().find(|x| !h.contains(x)) {
                return Err(bad_request("invalid_scope", format!("'{extra}' is not in the subject token")));
            }
            Some(w)
        }
        (h, None) => h,
        (None, w) => w,
    };
    let iss = cfg.self_issuer();
    // the prior chain: the subject token's own `act`, kept as issued
    let prior = sv.claims.get("act").cloned();
    let depth = schain.len() - 1 + usize::from(actor.is_some());
    if depth > MAX_CHAIN {
        return Err(bad_request("invalid_request", format!("actor chain deeper than {MAX_CHAIN}")));
    }
    let now = now_secs();
    let ttl = cfg.token_ttl_secs.unwrap_or(600);
    let mut claims = json!({
        "iss": iss, "sub": subject, "sub_profile": profile_of(&skind), "aud": cfg.resource,
        "iat": now, "exp": now + ttl, "jti": b64url(&rand::random::<[u8; 16]>()),
    });
    if let Some(s) = &scope {
        claims["scope"] = json!(s.join(" "));
    }
    for k in ["kleos_ws", "kleos_run", "client_id", "txn"] {
        if let Some(x) = sv.claims.get(k) {
            claims[k] = x.clone();
        }
    }
    if let Some(w) = form.get("workspace") {
        claims["kleos_ws"] = json!(w);
    }
    match (actor, prior) {
        (Some((id, kind)), prior) => {
            let mut act = json!({"sub": id, "iss": iss, "sub_profile": profile_of(&kind)});
            if let Some(p) = prior {
                act["act"] = p;
            }
            claims["act"] = act;
        }
        (None, Some(p)) => claims["act"] = p,
        (None, None) => {}
    }
    let mut token_type = "Bearer";
    if let Some(proof) = header("dpop") {
        claims["cnf"] = json!({"jkt": verify_dpop(&proof, req, None)?});
        token_type = "DPoP";
    }
    let token = encode(&Header::new(Algorithm::HS256), &claims, &EncodingKey::from_secret(&secret)).map_err(|e| bad_request("server_error", e.to_string()))?;
    let mut out = json!({
        "access_token": token,
        "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "token_type": token_type,
        "expires_in": ttl,
    });
    if let Some(s) = scope {
        out["scope"] = json!(s.join(" "));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    /// A P-256 key pair for tests (PKCS#8 PEM and its public JWK).
    const EC_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg+vC2PCRZiDZ6V5xL
vT6RXpHjcaCCUQBj0CY7nAeHpAChRANCAASalGaMeaFRXL0fe5aeX61b++/tgWwL
QJ97/AMw01C+60S2fOQHMUMyj3tQR7Tqbs+cKO8MsNTJ8asi5whuumGb
-----END PRIVATE KEY-----";
    fn ec_jwk() -> Value {
        json!({"kty": "EC", "crv": "P-256", "kid": "k1", "alg": "ES256",
               "x": "mpRmjHmhUVy9H3uWnl-tW_vv7YFsC0Cfe_wDMNNQvus", "y": "RLZ85AcxQzKPe1BHtOpuz5wo7wyw1MnxqyLnCG66YZs"})
    }
    const HS: &str = "s3cret-s3cret-s3cret-s3cret-s3cret";

    fn store() -> DbInstance {
        let db = DbInstance::new("mem", "", "").unwrap();
        db.tenant_call("org_create", json!({"org": "acme", "owner": "alice"})).unwrap();
        db.tenant_call(
            "principal_create",
            json!({"id": "bot", "org": "acme", "kind": "agent", "sponsor": "alice",
                   "idents": [["https://agents.example", "travel"], ["spiffe://example.org", "spiffe://example.org/agents/bot"]]}),
        )
        .unwrap();
        db
    }

    fn cfg() -> TenancyConfig {
        std::env::set_var("KLEOS_TEST_HS", HS);
        std::env::set_var("KLEOS_TEST_SIGN", "kleos-signing-key-kleos-signing-key-0001");
        TenancyConfig {
            resource: "https://kleos.example/mcp".into(),
            issuers: vec![IssuerConfig {
                iss: "https://idp.example".into(),
                alg: Some("HS256".into()),
                secret_env: Some("KLEOS_TEST_HS".into()),
                map: Some("direct".into()),
                ..Default::default()
            }],
            api_keys: true,
            signing_secret_env: Some("KLEOS_TEST_SIGN".into()),
            ..Default::default()
        }
    }

    fn exp() -> u64 {
        now_secs() + 600
    }

    fn token(claims: Value) -> String {
        encode(&Header::default(), &claims, &EncodingKey::from_secret(HS.as_bytes())).unwrap()
    }

    fn es256(claims: Value, header: Header) -> String {
        encode(&header, &claims, &EncodingKey::from_ec_pem(EC_PEM.as_bytes()).unwrap()).unwrap()
    }

    fn dpop(method: &str, url: &str, at: Option<&str>) -> String {
        let mut h = Header::new(Algorithm::ES256);
        h.typ = Some("dpop+jwt".into());
        h.jwk = Some(serde_json::from_value(ec_jwk()).unwrap());
        let mut c = json!({"htm": method, "htu": url, "iat": now_secs(), "jti": b64url(&rand::random::<[u8; 12]>())});
        if let Some(at) = at {
            c["ath"] = json!(b64url(&Sha256::digest(at.as_bytes())));
        }
        es256(c, h)
    }

    #[test]
    fn jwt_with_actor_chain() {
        let (db, cfg) = (store(), cfg());
        let t = token(json!({
            "iss": "https://idp.example", "sub": "alice", "sub_profile": "user", "aud": "https://kleos.example/mcp", "exp": exp(),
            "scope": "kleos:read kleos:write", "client_id": "https://app.example/client.json",
            "act": {"sub": "booking-tool", "iss": "https://as.example", "sub_profile": "service",
                    "act": {"sub": "travel", "iss": "https://agents.example", "sub_profile": "ai_agent"}}
        }));
        let ctx = authenticate(&cfg, &db, &t, &|_| None).unwrap();
        assert_eq!(ctx.human.as_deref(), Some("alice"));
        assert_eq!(ctx.agent.as_deref(), Some("bot"));
        let kinds: Vec<&str> = ctx.chain.iter().map(|l| l.kind.as_str()).collect();
        assert_eq!(kinds, ["human", "ai_agent", "service"]);
        assert_eq!(ctx.chain[0].assurance, Assurance::Verified);
        assert_eq!(ctx.chain[1].assurance, Assurance::IssuerEndorsed);
        assert_eq!(ctx.client_id.as_deref(), Some("https://app.example/client.json"));
        assert!(may_write(&ctx));
        // the wrong audience is refused (RFC 8707: tokens for this resource only)
        let bad = token(json!({"iss": "https://idp.example", "sub": "alice", "aud": "https://other.example", "exp": exp()}));
        assert_eq!(authenticate(&cfg, &db, &bad, &|_| None).unwrap_err().status, 401);
        // a chain deeper than the limit is an error, not truncated
        let mut act = json!({"sub": "a0"});
        for i in 1..8 {
            act = json!({"sub": format!("a{i}"), "act": act});
        }
        let deep = token(json!({"iss": "https://idp.example", "sub": "alice", "aud": "https://kleos.example/mcp", "exp": exp(), "act": act}));
        assert_eq!(authenticate(&cfg, &db, &deep, &|_| None).unwrap_err().error, "invalid_request");
    }

    #[test]
    fn api_key_with_claimed_human() {
        let (db, cfg) = (store(), cfg());
        let key = db.tenant_call("apikey_create", json!({"principal": "bot", "actions": ["read"]})).unwrap()["key"].as_str().unwrap().to_string();
        let ctx = authenticate(&cfg, &db, &key, &|h| (h == "x-kleos-on-behalf-of").then(|| "alice".to_string())).unwrap();
        assert_eq!(ctx.agent.as_deref(), Some("bot"));
        assert_eq!(ctx.chain[0].assurance, Assurance::SelfAsserted);
        assert_eq!(ctx.chain[1].assurance, Assurance::Verified);
        assert!(!may_write(&ctx));
        assert!(authenticate(&cfg, &db, "kl_nope", &|_| None).is_err());
        assert!(cfg.metadata()["resource"] == "https://kleos.example/mcp");
    }

    /// One-shot HTTP server on localhost answering `bodies` in order.
    fn serve(bodies: Vec<String>) -> (String, std::thread::JoinHandle<usize>) {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/jwks", l.local_addr().unwrap());
        let h = std::thread::spawn(move || {
            let mut n = 0;
            for b in bodies {
                let (mut s, _) = l.accept().unwrap();
                let mut buf = [0u8; 2048];
                let _ = s.read(&mut buf);
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                n += 1;
            }
            n
        });
        (url, h)
    }

    #[test]
    fn jwks_url_is_cached_and_refetched_on_rotation() {
        let db = store();
        let k1 = json!({"keys": [{"kty": "oct", "kid": "old", "k": b64url(b"first-key-first-key-first-key-01")}]});
        let k2 = json!({"keys": [{"kty": "oct", "kid": "new", "k": b64url(b"second-key-second-key-second-k02")}]});
        let (url, srv) = serve(vec![k1.to_string(), k2.to_string()]);
        let cfg = TenancyConfig {
            resource: "https://kleos.example/mcp".into(),
            issuers: vec![IssuerConfig { iss: "https://rot.example".into(), alg: Some("HS256".into()), jwks_uri: Some(url), map: Some("direct".into()), ..Default::default() }],
            ..Default::default()
        };
        let sign = |kid: &str, key: &[u8]| {
            let mut h = Header::default();
            h.kid = Some(kid.into());
            encode(&h, &json!({"iss": "https://rot.example", "sub": "alice", "aud": "https://kleos.example/mcp", "exp": exp()}), &EncodingKey::from_secret(key)).unwrap()
        };
        // fetched once, then served from the cache
        for _ in 0..3 {
            assert!(authenticate(&cfg, &db, &sign("old", b"first-key-first-key-first-key-01"), &|_| None).is_ok());
        }
        // a rotated key: one refetch
        assert!(authenticate(&cfg, &db, &sign("new", b"second-key-second-key-second-k02"), &|_| None).is_ok());
        assert_eq!(srv.join().unwrap(), 2);
    }

    #[test]
    fn spiffe_svid_txn_token_and_dpop() {
        let (db, mut cfg) = (store(), cfg());
        cfg.issuers.push(IssuerConfig { iss: "spiffe://example.org".into(), alg: Some("ES256".into()), jwks: Some(json!({"keys": [ec_jwk()]})), spiffe_trust_domain: Some("example.org".into()), ..Default::default() });
        cfg.issuers.push(IssuerConfig { iss: "https://tts.example".into(), alg: Some("HS256".into()), secret_env: Some("KLEOS_TEST_HS".into()), map: Some("direct".into()), txn: true, ..Default::default() });
        // a JWT-SVID (no iss): the agent itself, verified
        let mut h = Header::new(Algorithm::ES256);
        h.kid = Some("k1".into());
        let svid = es256(json!({"sub": "spiffe://example.org/agents/bot", "aud": "https://kleos.example/mcp", "exp": exp()}), h);
        let ctx = authenticate(&cfg, &db, &svid, &|_| None).unwrap();
        assert_eq!(ctx.agent.as_deref(), Some("bot"));
        assert_eq!(ctx.chain[0].assurance, Assurance::Verified);
        // a Transaction Token alone: subject, txn id, narrowed scope, workload
        let tt = token(json!({"iss": "https://tts.example", "sub": "alice", "aud": "https://kleos.example/mcp", "exp": exp(), "txn": "t-42", "scope": "kleos:read", "req_wl": "gateway"}));
        let ctx = authenticate_request(&cfg, &db, None, &|h| (h == "txn-token").then(|| tt.clone()), &RequestInfo::default()).unwrap();
        assert_eq!((ctx.human.as_deref(), ctx.txn.as_deref()), (Some("alice"), Some("t-42")));
        assert!(!may_write(&ctx));
        assert_eq!(ctx.chain.last().unwrap().sub, "gateway");
        // an access token kleos minted with a DPoP binding needs the proof
        let req = RequestInfo { method: "POST".into(), url: "https://kleos.example/oauth/token".into() };
        let user = token(json!({"iss": "https://idp.example", "sub": "alice", "aud": "https://kleos.example/mcp", "exp": exp(), "scope": "kleos:read kleos:write"}));
        db.tenant_call("principal_key_set", json!({"id": "bot", "jwks": {"keys": [ec_jwk()]}})).unwrap();
        let mut ah = Header::new(Algorithm::ES256);
        ah.kid = Some("k1".into());
        let assertion = es256(json!({"iss": "bot", "sub": "bot", "aud": req.url, "exp": exp(), "jti": "a1"}), ah);
        let form: HashMap<String, String> = [
            ("grant_type", "urn:ietf:params:oauth:grant-type:token-exchange"),
            ("subject_token", user.as_str()),
            ("client_assertion_type", "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"),
            ("client_assertion", assertion.as_str()),
            ("scope", "kleos:read"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let proof = dpop("POST", &req.url, None);
        let r = token_exchange(&cfg, &db, &form, &|h| (h == "dpop").then(|| proof.clone()), &req).unwrap();
        assert_eq!(r["token_type"], "DPoP");
        let at = r["access_token"].as_str().unwrap().to_string();
        let claims = payload_of(&at).unwrap();
        assert_eq!((claims["sub"].as_str(), claims["act"]["sub"].as_str()), (Some("alice"), Some("bot")));
        // used without a proof: refused; with a proof for the request: alice through bot
        let mem = RequestInfo { method: "POST".into(), url: "https://kleos.example/memory/recall".into() };
        assert_eq!(authenticate_request(&cfg, &db, Some(&at), &|_| None, &mem).unwrap_err().error, "invalid_dpop_proof");
        let p2 = dpop("POST", &mem.url, Some(&at));
        let ctx = authenticate_request(&cfg, &db, Some(&at), &|h| (h == "dpop").then(|| p2.clone()), &mem).unwrap();
        assert_eq!((ctx.human.as_deref(), ctx.agent.as_deref()), (Some("alice"), Some("bot")));
        assert!(!may_write(&ctx), "scopes only narrow");
        // the same proof again is a replay
        assert!(authenticate_request(&cfg, &db, Some(&at), &|h| (h == "dpop").then(|| p2.clone()), &mem).is_err());
        // asking for a scope the subject token lacks is refused
        let mut f2 = form.clone();
        f2.insert("scope".into(), "kleos:admin".into());
        f2.insert("client_assertion".into(), {
            let mut ah = Header::new(Algorithm::ES256);
            ah.kid = Some("k1".into());
            es256(json!({"iss": "bot", "sub": "bot", "aud": req.url, "exp": exp(), "jti": "a2"}), ah)
        });
        assert_eq!(token_exchange(&cfg, &db, &f2, &|_| None, &req).unwrap_err().error, "invalid_scope");
    }
}
