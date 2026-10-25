/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Credentials of principal-mode requests (docs/specs/tenancy.md,
//! "Surfaces"): RFC 9068 JWT access tokens with OAuth Actor Profile `act`
//! chains, per-principal API keys, and RFC 9728 protected-resource
//! metadata. kleos is a resource server: it validates the audience and never
//! forwards an inbound token.

use base64::Engine;
use jsonwebtoken::jwk::JwkSet;
use jsonwebtoken::{decode, decode_header, Algorithm, DecodingKey, Validation};
use kleos::tenancy::{ActorLink, Assurance, PrincipalContext, TokenGrant, MAX_CHAIN};
use kleos::DbInstance;
use serde_derive::Deserialize;
use serde_json::{json, Value};

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

impl TenancyConfig {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("bad tenancy config {path}: {e}"))
    }

    /// RFC 9728 protected-resource metadata.
    pub fn metadata(&self) -> Value {
        json!({
            "resource": self.resource,
            "authorization_servers": self.authorization_servers,
            "bearer_methods_supported": ["header"],
            "scopes_supported": ["kleos:read", "kleos:write", "kleos:promote", "kleos:admin"],
            "resource_name": "kleos",
            "resource_documentation": "https://github.com/kleos-db/kleos/blob/main/docs/specs/tenancy.md",
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

/// The kleos principal behind `(iss, sub)`.
fn map_principal(db: &DbInstance, issuer: Option<&IssuerConfig>, iss: &str, sub: &str) -> Option<String> {
    if let Ok(Some(p)) = db.resolve_ident(iss, sub) {
        return Some(p);
    }
    let direct = issuer.is_some_and(|i| i.map.as_deref() == Some("direct"));
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

/// Authenticate a bearer credential into a principal context. `header`
/// reads request headers (`X-Kleos-Workspace`, `X-Kleos-Run`,
/// `X-Kleos-On-Behalf-Of`).
pub fn authenticate(cfg: &TenancyConfig, db: &DbInstance, bearer: &str, header: &dyn Fn(&str) -> Option<String>) -> Result<PrincipalContext, AuthError> {
    let workspace = header("x-kleos-workspace").or_else(|| cfg.default_workspace.clone());
    let run = header("x-kleos-run");
    if bearer.starts_with("kl_") {
        if !cfg.api_keys {
            return Err(unauthorized("API keys are disabled"));
        }
        let (p, actions) = db.resolve_apikey(bearer).ok().flatten().ok_or_else(|| unauthorized("unknown or expired API key"))?;
        let kind = db.principal_kind(&p).ok().flatten().ok_or_else(|| unauthorized("the key's principal is not active"))?;
        let mut ctx = PrincipalContext { workspace, run, ..Default::default() };
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
        return Ok(ctx);
    }
    // a JWT access token: find the issuer, then verify
    let parts: Vec<&str> = bearer.split('.').collect();
    if parts.len() != 3 {
        return Err(unauthorized("not a JWT or an API key"));
    }
    let unverified: Value = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts[1].trim_end_matches('='))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .ok_or_else(|| unauthorized("malformed token"))?;
    let iss = unverified.get("iss").and_then(|v| v.as_str()).ok_or_else(|| unauthorized("token has no iss"))?.to_string();
    let issuer = cfg.issuer(&iss).ok_or_else(|| unauthorized(format!("untrusted issuer '{iss}'")))?;
    let head = decode_header(bearer).map_err(|e| unauthorized(format!("bad token header: {e}")))?;
    let want_alg = match issuer.alg.as_deref().unwrap_or("RS256") {
        "RS256" => Algorithm::RS256,
        "ES256" => Algorithm::ES256,
        "EdDSA" => Algorithm::EdDSA,
        "HS256" => Algorithm::HS256,
        other => return Err(unauthorized(format!("unsupported alg '{other}' configured"))),
    };
    if head.alg != want_alg {
        return Err(unauthorized("unexpected token algorithm"));
    }
    let key = if let Some(env) = &issuer.secret_env {
        let secret = std::env::var(env).map_err(|_| unauthorized("issuer secret is not set"))?;
        DecodingKey::from_secret(secret.as_bytes())
    } else if let Some(jwks) = &issuer.jwks {
        let set: JwkSet = serde_json::from_value(jwks.clone()).map_err(|e| unauthorized(format!("bad JWKS: {e}")))?;
        let jwk = match &head.kid {
            Some(kid) => set.find(kid),
            None => set.keys.first(),
        }
        .ok_or_else(|| unauthorized("no matching key"))?;
        DecodingKey::from_jwk(jwk).map_err(|e| unauthorized(format!("bad key: {e}")))?
    } else if let Some(pem) = &issuer.pem {
        match want_alg {
            Algorithm::ES256 => DecodingKey::from_ec_pem(pem.as_bytes()),
            Algorithm::EdDSA => DecodingKey::from_ed_pem(pem.as_bytes()),
            _ => DecodingKey::from_rsa_pem(pem.as_bytes()),
        }
        .map_err(|e| unauthorized(format!("bad key: {e}")))?
    } else {
        return Err(unauthorized("issuer has no key configured"));
    };
    let mut v = Validation::new(want_alg);
    v.set_audience(&[&cfg.resource]);
    v.set_issuer(&[&iss]);
    v.leeway = 30;
    let claims: Value = decode::<Value>(bearer, &key, &v).map_err(|e| unauthorized(format!("token rejected: {e}")))?.claims;
    let sub = claims.get("sub").and_then(|v| v.as_str()).ok_or_else(|| unauthorized("token has no sub"))?.to_string();
    // the chain, innermost (the subject) first: the nested `act` objects
    // run from the presenter (outermost) inwards
    let mut acts = vec![];
    let mut cur = claims.get("act");
    while let Some(a) = cur {
        if acts.len() >= MAX_CHAIN {
            return Err(AuthError { status: 400, error: "invalid_request", description: format!("actor chain deeper than {MAX_CHAIN}") });
        }
        acts.push(a.clone());
        cur = a.get("act");
    }
    let sub_principal = map_principal(db, Some(issuer), &iss, &sub);
    let sub_kind = sub_principal.as_deref().and_then(|p| db.principal_kind(p).ok().flatten());
    let mut chain = vec![ActorLink {
        kind: link_kind(claims.get("sub_profile").and_then(|v| v.as_str()), sub_kind.as_deref()),
        iss: Some(iss.clone()),
        sub: sub.clone(),
        principal: sub_principal.clone(),
        assurance: Assurance::Verified,
    }];
    for a in acts.iter().rev() {
        let (Some(asub), aiss) = (a.get("sub").and_then(|v| v.as_str()), a.get("iss").and_then(|v| v.as_str()).unwrap_or(&iss)) else {
            return Err(AuthError { status: 400, error: "invalid_request", description: "an act object needs sub".into() });
        };
        let p = map_principal(db, cfg.issuer(aiss).or(Some(issuer)), aiss, asub);
        let pk = p.as_deref().and_then(|x| db.principal_kind(x).ok().flatten());
        chain.push(ActorLink {
            kind: link_kind(a.get("sub_profile").and_then(|v| v.as_str()), pk.as_deref()),
            iss: Some(aiss.to_string()),
            sub: asub.to_string(),
            principal: p,
            assurance: Assurance::IssuerEndorsed,
        });
    }
    let Some(sp) = sub_principal else {
        return Err(AuthError { status: 403, error: "insufficient_scope", description: format!("subject '{sub}' of '{iss}' is not a kleos principal") });
    };
    let mut ctx = PrincipalContext {
        workspace: workspace.or_else(|| claims.get("kleos_ws").and_then(|v| v.as_str()).map(String::from)),
        run: run.or_else(|| claims.get("kleos_run").and_then(|v| v.as_str()).map(String::from)),
        token: grant_of(claims.get("scope").and_then(|v| v.as_str())),
        client_id: claims.get("client_id").and_then(|v| v.as_str()).map(String::from),
        txn: claims.get("txn").and_then(|v| v.as_str()).map(String::from),
        ..Default::default()
    };
    match chain[0].kind.as_str() {
        "ai_agent" => ctx.agent = Some(sp),
        "service" => ctx.service = Some(sp),
        _ => ctx.human = Some(sp),
    }
    // the acting agent: the outermost agent link that maps to a principal
    if let Some(a) = chain.iter().skip(1).rev().find(|l| l.kind == "ai_agent" && l.principal.is_some()) {
        ctx.agent = a.principal.clone();
    }
    ctx.chain = chain;
    Ok(ctx)
}

/// Whether a context's credential may write.
pub fn may_write(ctx: &PrincipalContext) -> bool {
    ctx.token.as_ref().and_then(|t| t.actions.as_ref()).map_or(true, |a| a.iter().any(|x| x == "write" || x == "promote" || x == "admin"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{encode, EncodingKey, Header};

    fn store() -> DbInstance {
        let db = DbInstance::new("mem", "", "").unwrap();
        db.tenant_call("org_create", json!({"org": "acme", "owner": "alice"})).unwrap();
        db.tenant_call("principal_create", json!({"id": "bot", "org": "acme", "kind": "agent", "sponsor": "alice", "idents": [["https://agents.example", "travel"]]}))
            .unwrap();
        db
    }

    fn cfg() -> TenancyConfig {
        std::env::set_var("KLEOS_TEST_HS", "s3cret-s3cret-s3cret-s3cret-s3cret");
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
            ..Default::default()
        }
    }

    fn token(claims: Value) -> String {
        encode(&Header::default(), &claims, &EncodingKey::from_secret(b"s3cret-s3cret-s3cret-s3cret-s3cret")).unwrap()
    }

    #[test]
    fn jwt_with_actor_chain() {
        let (db, cfg) = (store(), cfg());
        let exp = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() + 600) as i64;
        let t = token(json!({
            "iss": "https://idp.example", "sub": "alice", "sub_profile": "user", "aud": "https://kleos.example/mcp", "exp": exp,
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
        let bad = token(json!({"iss": "https://idp.example", "sub": "alice", "aud": "https://other.example", "exp": exp}));
        assert_eq!(authenticate(&cfg, &db, &bad, &|_| None).unwrap_err().status, 401);
        // a chain deeper than the limit is an error, not truncated
        let mut act = json!({"sub": "a0"});
        for i in 1..8 {
            act = json!({"sub": format!("a{i}"), "act": act});
        }
        let deep = token(json!({"iss": "https://idp.example", "sub": "alice", "aud": "https://kleos.example/mcp", "exp": exp, "act": act}));
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
}
