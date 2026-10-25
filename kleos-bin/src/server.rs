/*
 * Copyright 2023, The Cozo Project Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::path::{Path as FsPath, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{header, HeaderName, HeaderValue, Method, Request, Response, StatusCode};
use axum::response::sse::{Event, KeepAlive};
use axum::response::{Html, Sse};
use axum::routing::{get, post, put};
use axum::{Extension, Json, Router};
use clap::Args;
use futures::future::BoxFuture;
use futures::stream::Stream;
use itertools::Itertools;
use log::{error, info, warn};
use miette::miette;
// use miette::miette;
use rand::Rng;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use tokio::task::spawn_blocking;
use tower_http::auth::{AsyncAuthorizeRequest, AsyncRequireAuthorizationLayer};
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};

use kleos::{
    format_error_as_json, DataValue, DbInstance, MultiTransaction, NamedRows, ScriptMutability,
    ScriptRunOptions, SimpleFixedRule, SubscribeOptions, SubscriptionEvent,
};

#[derive(Args, Debug)]
pub(crate) struct ServerArgs {
    /// Database engine, can be `mem`, `sqlite`, `rocksdb` and others.
    #[clap(short, long, default_value_t = String::from("mem"))]
    engine: String,

    /// Path to the directory to store the database
    #[clap(short, long, default_value_t = String::from("cozo.db"))]
    path: String,

    /// Directory containing files exposed by the HTTP backup APIs
    #[clap(long, default_value_t = String::from("backups"))]
    backup_dir: String,

    /// Restore from the specified backup before starting the server
    #[clap(long)]
    restore: Option<String>,

    /// Extra config in JSON format
    #[clap(short, long, default_value_t = String::from("{}"))]
    config: String,

    /// Address to bind the service to
    #[clap(short, long, default_value_t = String::from("127.0.0.1"))]
    bind: String,

    /// Disable HTTP authentication. This is intentionally explicit and is
    /// accepted only for a loopback bind address.
    #[clap(long, default_value_t = false)]
    insecure_no_auth: bool,

    /// Port to use
    #[clap(short = 'P', long, default_value_t = 9070)]
    port: u16,

    /// When set, the content of the named table will be used as a token table
    #[clap(long)]
    token_table: Option<String>,

    /// Default per-query wall-clock budget in seconds (kleos fork, query
    /// budget). Unset or non-positive = no default. Bounds every query; a
    /// per-request `timeout` or an in-script `:timeout` can only tighten it.
    #[clap(long)]
    default_query_timeout: Option<f64>,

    /// Db-wide default per-query memory budget in estimated bytes (kleos
    /// fork, query memory budget; spec `docs/specs/memory-budget.md`).
    /// kleos-bin is an application, not a library, so it ARMS a budget out of
    /// the box (4 GiB); pass `0` for unlimited. A per-request `mem_limit` or
    /// an in-script `:mem_limit` can only tighten it.
    #[clap(long, default_value_t = 4 * 1024 * 1024 * 1024)]
    default_query_mem_limit: u64,

    /// Turn a built-in embedder on at startup (repeatable):
    /// `--embedder name=model[:precision[:device[:dim]]]`, e.g.
    /// `--embedder gemma=embeddinggemma:int8` or
    /// `--embedder potion=potion-base-8M`. Needs a build with the `infer`
    /// (or `infer-metal`) feature. `::embedder enable/disable` switch
    /// embedders at runtime too, also over HTTP.
    #[clap(long = "embedder")]
    embedders: Vec<String>,
    /// Run built-in durable workflows and schedules in this process
    /// (docs/specs/durable-workflows.md): the queues to serve, e.g.
    /// `--flow-worker default`. Repeat for several queues.
    #[clap(long = "flow-worker")]
    flow_queues: Vec<String>,
    /// Also serve MCP (protocol 2026-07-28, stateless Streamable HTTP) at
    /// `POST /mcp`, behind the same authentication
    /// (docs/specs/mcp-cli-skills.md).
    #[clap(long, default_value_t = false)]
    mcp: bool,
    /// Mirror a Postgres source (kleos-pg, feature `pg-cdc`;
    /// docs/specs/pg-connector.md): the connector's YAML config. Repeat for
    /// several sources. Each runs as a task of this server.
    #[clap(long = "pg-cdc")]
    pg_cdc: Vec<String>,
    /// Default namespace of the MCP tools.
    #[clap(long, default_value_t = String::from("default"))]
    mcp_ns: String,
    /// Offer the read-only `query` tool over MCP.
    #[clap(long, default_value_t = false)]
    mcp_allow_query: bool,
    /// Principal mode (docs/specs/tenancy.md): a JSON config naming this
    /// resource and the trusted token issuers. `/memory/:op` and `/mcp`
    /// then accept JWT access tokens and API keys and run as that principal;
    /// `/.well-known/oauth-protected-resource` serves RFC 9728 metadata.
    #[clap(long)]
    tenancy: Option<String>,
    /// What MCP `skills/list` returns without a task: `top` (the most useful
    /// live skills, a partial listing), `full` (the catalog, paged) or `none`
    /// (docs/specs/skills.md).
    #[clap(long, default_value_t = String::from("top"))]
    mcp_skills_list: String,
}

#[derive(Clone)]
struct DbState {
    db: DbInstance,
    tenancy: Option<Arc<kleos_bin::auth::TenancyConfig>>,
    mcp: Option<Arc<kleos_bin::mcp::McpServer>>,
    origin: Arc<String>,
    backup_dir: Arc<PathBuf>,
    rule_senders: Arc<Mutex<BTreeMap<u32, crossbeam::channel::Sender<miette::Result<NamedRows>>>>>,
    rule_counter: Arc<AtomicU32>,
    tx_counter: Arc<AtomicU32>,
    txs: Arc<Mutex<BTreeMap<u32, StoredTransaction>>>,
}

struct StoredTransaction {
    tx: Arc<MultiTransaction>,
    owner: Principal,
    write: bool,
    expires_at: Instant,
    in_flight: usize,
}

const TRANSACTION_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
const TRANSACTION_REAPER_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Debug, Eq, PartialEq)]
enum Principal {
    Local,
    Admin,
    Bearer([u8; 32]),
    /// A tenancy principal (its id).
    Tenant(String),
}

#[derive(Clone)]
struct AuthContext {
    mutability: ScriptMutability,
    principal: Principal,
    /// Principal mode: the authenticated principal context.
    tenant: Option<Arc<kleos::tenancy::PrincipalContext>>,
}

struct TransactionLease {
    txs: Arc<Mutex<BTreeMap<u32, StoredTransaction>>>,
    id: u32,
}

impl Drop for TransactionLease {
    fn drop(&mut self) {
        if let Some(stored) = self.txs.lock().unwrap().get_mut(&self.id) {
            stored.in_flight = stored.in_flight.saturating_sub(1);
            if stored.in_flight == 0 {
                stored.expires_at = Instant::now() + TRANSACTION_IDLE_TIMEOUT;
            }
        }
    }
}

fn bearer_principal(token: &str) -> Principal {
    Principal::Bearer(Sha256::digest(token.as_bytes()).into())
}

fn is_idle_expired(expires_at: Instant, in_flight: usize, now: Instant) -> bool {
    in_flight == 0 && expires_at <= now
}

#[derive(Clone)]
struct MyAuth {
    skip_auth: bool,
    allowed_origin: String,
    auth_guard: String,
    token_table: Option<Arc<(String, DbInstance)>>,
    /// Principal mode: the config and the store that resolves credentials.
    tenancy: Option<Arc<(kleos_bin::auth::TenancyConfig, DbInstance)>>,
}

/// Paths a tenancy principal may reach; everything else (raw scripts,
/// import/export, backups) stays with the admin secret.
fn tenant_path_ok(path: &str) -> bool {
    path.starts_with("/memory/") || path == "/mcp" || path.starts_with("/tenant/")
}

fn has_forbidden_insecure_request(request: &Request<Body>, allowed_origin: &str) -> bool {
    let Some(allowed_authority) = allowed_origin.strip_prefix("http://") else {
        return true;
    };
    let request_authority = request
        .headers()
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .or_else(|| request.uri().authority().map(|value| value.as_str()));
    if request_authority != Some(allowed_authority) {
        return true;
    }

    if request
        .headers()
        .get("sec-fetch-site")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("cross-site"))
    {
        return true;
    }

    request
        .headers()
        .get(header::ORIGIN)
        .is_some_and(|origin| origin.as_bytes() != allowed_origin.as_bytes())
}

impl AsyncAuthorizeRequest<Body> for MyAuth {
    type RequestBody = Body;
    type ResponseBody = Body;
    type Future = BoxFuture<'static, Result<Request<Body>, Response<Self::ResponseBody>>>;

    fn authorize(&mut self, mut request: Request<Body>) -> Self::Future {
        let skip_auth = self.skip_auth;
        let allowed_origin = self.allowed_origin.clone();
        let auth_guard = self.auth_guard.clone();
        let token_table = self.token_table.clone();
        let tenancy = self.tenancy.clone();
        Box::pin(async move {
            // the token endpoint authenticates its own subject and actor tokens
            if tenancy.is_some() && request.uri().path() == "/oauth/token" {
                request.extensions_mut().insert(ScriptMutability::Immutable);
                request.extensions_mut().insert(AuthContext { mutability: ScriptMutability::Immutable, principal: Principal::Local, tenant: None });
                return Ok(request);
            }
            if skip_auth {
                // Loopback is not an authentication boundary for browsers. Reject
                // requests originating on another site before granting the
                // unauthenticated local-server exception.
                if has_forbidden_insecure_request(&request, &allowed_origin) {
                    return Err(Response::builder()
                        .status(StatusCode::FORBIDDEN)
                        .body(Body::empty())
                        .unwrap());
                }
                request.extensions_mut().insert(ScriptMutability::Mutable);
                request.extensions_mut().insert(AuthContext {
                    mutability: ScriptMutability::Mutable,
                    principal: Principal::Local,
                    tenant: None,
                });
                return Ok(request);
            }

            // principal mode: a JWT access token or an API key (never the
            // admin secret, which keeps its own meaning)
            let bearer = request
                .headers()
                .get("Authorization")
                .and_then(|h| h.to_str().ok())
                .and_then(|h| h.strip_prefix("Bearer ").or_else(|| h.strip_prefix("DPoP ")))
                .map(|t| t.trim().to_string());
            let has_txn = request.headers().contains_key("txn-token");
            let tok = bearer.clone().unwrap_or_default();
            if let (Some(tn), true) = (&tenancy, bearer.is_some() || has_txn) {
                if auth_guard.is_empty() || tok != auth_guard {
                    let (cfg, db) = (tn.0.clone(), tn.1.clone());
                    let headers = request.headers().clone();
                    let t2 = bearer.clone();
                    let info = kleos_bin::auth::RequestInfo {
                        method: request.method().as_str().to_string(),
                        url: format!("{}{}", allowed_origin.trim_end_matches('/'), request.uri().path()),
                    };
                    let res = spawn_blocking(move || {
                        let h = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).map(|x| x.to_string());
                        kleos_bin::auth::authenticate_request(&cfg, &db, t2.as_deref(), &h, &info)
                    })
                    .await
                    .unwrap_or_else(|e| Err(kleos_bin::auth::AuthError { status: 500, error: "server_error", description: e.to_string() }));
                    match res {
                        Ok(ctx) => {
                            if !tenant_path_ok(request.uri().path()) {
                                return Err(Response::builder().status(StatusCode::FORBIDDEN).body(Body::empty()).unwrap());
                            }
                            let mutability =
                                if kleos_bin::auth::may_write(&ctx) { ScriptMutability::Mutable } else { ScriptMutability::Immutable };
                            let id = ctx.human.clone().or(ctx.agent.clone()).or(ctx.service.clone()).unwrap_or_default();
                            request.extensions_mut().insert(mutability);
                            request.extensions_mut().insert(AuthContext { mutability, principal: Principal::Tenant(id), tenant: Some(Arc::new(ctx)) });
                            return Ok(request);
                        }
                        // a token-table token is not a tenancy credential: fall through
                        Err(e) if token_table.is_none() || has_txn || tok.starts_with("kl_") || tok.matches('.').count() == 2 => {
                            return Err(Response::builder()
                                .status(StatusCode::from_u16(e.status).unwrap_or(StatusCode::UNAUTHORIZED))
                                .header(header::WWW_AUTHENTICATE, tn.0.challenge(&allowed_origin, Some(&e)))
                                .body(Body::empty())
                                .unwrap());
                        }
                        Err(_) => {}
                    }
                }
            }

            // Token-table bearer auth, evaluated INDEPENDENTLY of any query string.
            // Previously this lived only in the `query() == None` branch, so a
            // valid `Authorization: Bearer ...` was ignored for every endpoint
            // that needs query params (e.g. `/transact?write=true`,
            // `/rules/name?arity=2`). Returns None when there's no token table or
            // no usable bearer header, so callers can fall through to deny.
            let bearer_grant = || -> Option<AuthContext> {
                let tt = token_table.as_ref()?;
                let (name, db) = tt.as_ref();
                let auth_str = request.headers().get("Authorization")?.to_str().ok()?;
                let token = auth_str.strip_prefix("Bearer ")?;
                match db.run_script(
                    &format!("?[mutable] := *{name} {{ token: $token, mutable }}"),
                    BTreeMap::from([(String::from("token"), DataValue::from(token))]),
                    ScriptMutability::Immutable,
                ) {
                    Ok(rows) => rows.rows.first().map(|val| AuthContext {
                        mutability: if val[0].get_bool() == Some(true) {
                            ScriptMutability::Mutable
                        } else {
                            ScriptMutability::Immutable
                        },
                        principal: bearer_principal(token),
                        tenant: None,
                    }),
                    Err(err) => {
                        eprintln!("Error: {}", err);
                        None
                    }
                }
            };

            let mutability = match request.headers().get("x-cozo-auth") {
                None => {
                    // Try the `?auth=<secret>` query-string credential first, then
                    // fall back to bearer-token auth regardless of the query string.
                    let query_grant = request.uri().query().and_then(|q_str| {
                        for pair in q_str.split('&') {
                            if let Some((k, v)) = pair.split_once('=') {
                                if k == "auth" {
                                    return (v == auth_guard.as_str()).then_some(AuthContext {
                                        mutability: ScriptMutability::Mutable,
                                        principal: Principal::Admin,
                                        tenant: None,
                                    });
                                }
                            }
                        }
                        None
                    });
                    // the server's own secret as a bearer token (MCP clients
                    // send `Authorization: Bearer`, docs/specs/mcp-cli-skills.md)
                    let admin_bearer = request
                        .headers()
                        .get("Authorization")
                        .and_then(|h| h.to_str().ok())
                        .and_then(|h| h.strip_prefix("Bearer "))
                        .filter(|t| !auth_guard.is_empty() && *t == auth_guard.as_str())
                        .map(|_| AuthContext { mutability: ScriptMutability::Mutable, principal: Principal::Admin, tenant: None });
                    query_grant.or(admin_bearer).or_else(bearer_grant)
                }
                Some(data) => match data.to_str() {
                    Ok(s) => {
                        if s == auth_guard.as_str() {
                            Some(AuthContext {
                                mutability: ScriptMutability::Mutable,
                                principal: Principal::Admin,
                                tenant: None,
                            })
                        } else {
                            None
                        }
                    }
                    Err(_) => None,
                },
            };
            if let Some(auth) = mutability {
                request.extensions_mut().insert(auth.mutability);
                request.extensions_mut().insert(auth);
                Ok(request)
            } else {
                let mut unauthorized_response = Response::builder().status(StatusCode::UNAUTHORIZED);
                if let Some(tn) = &tenancy {
                    unauthorized_response = unauthorized_response.header(header::WWW_AUTHENTICATE, tn.0.challenge(&allowed_origin, None));
                }
                let unauthorized_response = unauthorized_response.body(Body::empty()).unwrap();

                Err(unauthorized_response)
            }
        })
    }
}

fn require_mutable(
    mutability: ScriptMutability,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    match mutability {
        ScriptMutability::Mutable => Ok(()),
        ScriptMutability::Immutable => Err((
            StatusCode::FORBIDDEN,
            json!({"ok": false, "message": "this operation requires a mutable token"}).into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutable_operations_require_mutable_authorization() {
        assert!(require_mutable(ScriptMutability::Mutable).is_ok());

        let (status, _) = require_mutable(ScriptMutability::Immutable).unwrap_err();
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    #[test]
    fn bearer_principals_are_stable_without_retaining_tokens() {
        assert_eq!(bearer_principal("secret"), bearer_principal("secret"));
        assert_ne!(bearer_principal("secret"), bearer_principal("other"));
        assert!(matches!(bearer_principal("secret"), Principal::Bearer(_)));
    }

    #[test]
    fn active_transactions_are_not_idle_expired() {
        let now = Instant::now();
        let expired = now.checked_sub(Duration::from_secs(1)).unwrap();
        assert!(is_idle_expired(expired, 0, now));
        assert!(!is_idle_expired(expired, 1, now));
    }

    #[test]
    fn loopback_request_guard_allows_non_browser_and_same_origin_requests() {
        let no_origin = Request::builder()
            .header(header::HOST, "127.0.0.1:9070")
            .body(Body::empty())
            .unwrap();
        assert!(!has_forbidden_insecure_request(
            &no_origin,
            "http://127.0.0.1:9070"
        ));

        let same_origin = Request::builder()
            .header(header::HOST, "127.0.0.1:9070")
            .header(header::ORIGIN, "http://127.0.0.1:9070")
            .body(Body::empty())
            .unwrap();
        assert!(!has_forbidden_insecure_request(
            &same_origin,
            "http://127.0.0.1:9070"
        ));

        let http2_authority = Request::builder()
            .uri("http://127.0.0.1:9070/")
            .body(Body::empty())
            .unwrap();
        assert!(!has_forbidden_insecure_request(
            &http2_authority,
            "http://127.0.0.1:9070"
        ));
    }

    #[test]
    fn loopback_request_guard_rejects_cross_origin_requests() {
        let request = Request::builder()
            .header(header::HOST, "127.0.0.1:9070")
            .header(header::ORIGIN, "https://attacker.example")
            .body(Body::empty())
            .unwrap();
        assert!(has_forbidden_insecure_request(
            &request,
            "http://127.0.0.1:9070"
        ));
    }

    #[test]
    fn loopback_request_guard_rejects_foreign_or_missing_authority() {
        let foreign_host = Request::builder()
            .header(header::HOST, "rebind.test:9070")
            .body(Body::empty())
            .unwrap();
        assert!(has_forbidden_insecure_request(
            &foreign_host,
            "http://127.0.0.1:9070"
        ));

        let missing_host = Request::new(Body::empty());
        assert!(has_forbidden_insecure_request(
            &missing_host,
            "http://127.0.0.1:9070"
        ));
    }

    #[test]
    fn loopback_request_guard_rejects_cross_site_fetch_metadata() {
        let request = Request::builder()
            .header(header::HOST, "127.0.0.1:9070")
            .header(header::ORIGIN, "http://127.0.0.1:9070")
            .header("sec-fetch-site", "cross-site")
            .body(Body::empty())
            .unwrap();
        assert!(has_forbidden_insecure_request(
            &request,
            "http://127.0.0.1:9070"
        ));
    }

    #[test]
    fn loopback_request_guard_accepts_ipv6_authority() {
        let request = Request::builder()
            .header(header::HOST, "[::1]:9070")
            .body(Body::empty())
            .unwrap();
        assert!(!has_forbidden_insecure_request(
            &request,
            "http://[::1]:9070"
        ));
    }
}

pub(crate) async fn server_main(args: ServerArgs) {
    let db = DbInstance::new(&args.engine, &args.path, &args.config).unwrap();
    std::fs::create_dir_all(&args.backup_dir).expect("Cannot create HTTP backup directory");
    let backup_dir =
        std::fs::canonicalize(&args.backup_dir).expect("Cannot resolve HTTP backup directory");
    for arg in &args.embedders {
        let (name, spec) = kleos::EmbedderSpec::parse_cli(arg)
            .and_then(|(n, s)| kleos::enable_embedder(&n, s.clone()).map(|_| (n, s)))
            .unwrap_or_else(|e| panic!("--embedder {arg}: {e:?}"));
        eprintln!("embedder '{name}' enabled ({spec:?})");
    }
    // kept alive for the server's lifetime
    let _flow_worker = (!args.flow_queues.is_empty()).then(|| {
        eprintln!("flow worker serving queues {:?}", args.flow_queues);
        db.flow_worker(kleos::FlowWorkerOpts {
            queues: args.flow_queues.clone(),
            ..Default::default()
        })
    });
    start_pg(&args.pg_cdc, &db);
    if args.default_query_mem_limit > 0 {
        db.set_default_query_mem_limit(Some(args.default_query_mem_limit as usize));
    }
    if let Some(secs) = args.default_query_timeout {
        db.set_default_query_timeout(Some(secs));
    }
    if let Some(p) = &args.restore {
        if let Err(err) = db.restore_backup(p) {
            error!("{}", err);
            error!("Restore from backup failed, terminate");
            panic!()
        }
    }

    let bind_is_loopback = IpAddr::from_str(&args.bind)
        .map(|address| address.is_loopback())
        .unwrap_or(false);
    if args.insecure_no_auth && !bind_is_loopback {
        panic!("--insecure-no-auth is permitted only with a loopback bind address");
    }
    let skip_auth = args.insecure_no_auth;
    let server_origin = if Ipv6Addr::from_str(&args.bind).is_ok() {
        format!("http://[{}]:{}", args.bind, args.port)
    } else {
        format!("http://{}:{}", args.bind, args.port)
    };

    let conf_path = if skip_auth {
        "".to_string()
    } else {
        format!("{}.{}.cozo_auth", args.path, args.engine)
    };
    let auth_guard = if skip_auth {
        "".to_string()
    } else {
        match tokio::fs::read_to_string(&conf_path).await {
            Ok(s) => s.trim().to_string(),
            Err(_) => {
                let s = rand::thread_rng()
                    .sample_iter(&rand::distributions::Alphanumeric)
                    .take(64)
                    .map(char::from)
                    .collect();
                tokio::fs::write(&conf_path, &s).await.unwrap();
                s
            }
        }
    };

    let tenancy: Option<kleos_bin::auth::TenancyConfig> = args.tenancy.as_ref().map(|p| {
        let c = kleos_bin::auth::TenancyConfig::load(p).unwrap_or_else(|e| panic!("--tenancy: {e}"));
        db.tenant_call("enable", serde_json::Value::Null).unwrap_or_else(|e| panic!("--tenancy: {e:?}"));
        eprintln!("principal mode: resource {} ({} issuers)", c.resource, c.issuers.len());
        c
    });
    let metadata = Arc::new(tenancy.as_ref().map(|c| c.metadata()));
    let auth_obj = MyAuth {
        skip_auth,
        allowed_origin: server_origin.clone(),
        auth_guard,
        token_table: args.token_table.map(|t| Arc::new((t, db.clone()))),
        tenancy: tenancy.clone().map(|c| Arc::new((c, db.clone()))),
    };

    let mcp = args.mcp.then(|| {
        eprintln!("MCP {} (stateless Streamable HTTP) at {}/mcp", kleos_bin::mcp::MODERN, server_origin);
        Arc::new(kleos_bin::mcp::McpServer::new(
            kleos_bin::backend::Backend::Embedded(db.clone()),
            args.mcp_ns.clone(),
            args.mcp_allow_query,
        ).with_skills_list(&args.mcp_skills_list))
    });
    let state = DbState {
        tenancy: tenancy.clone().map(Arc::new),
        mcp,
        origin: Arc::new(server_origin.clone()),
        db,
        backup_dir: Arc::new(backup_dir),
        rule_senders: Default::default(),
        rule_counter: Default::default(),
        tx_counter: Default::default(),
        txs: Default::default(),
    };
    let transaction_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(TRANSACTION_REAPER_INTERVAL);
        loop {
            interval.tick().await;
            let now = Instant::now();
            let expired = {
                let mut txs = transaction_state.txs.lock().unwrap();
                let expired_ids = txs
                    .iter()
                    .filter_map(|(&id, entry)| {
                        is_idle_expired(entry.expires_at, entry.in_flight, now).then_some(id)
                    })
                    .collect_vec();
                expired_ids
                    .into_iter()
                    .filter_map(|id| txs.remove(&id))
                    .collect_vec()
            };
            for entry in expired {
                // Aborting waits for the database worker, so never do it on the
                // async runtime thread. Removal first also prevents new work
                // from extending an already-expired transaction.
                tokio::task::spawn_blocking(move || {
                    if let Err(err) = entry.tx.abort() {
                        warn!("Failed to abort expired HTTP transaction: {err}");
                    }
                });
            }
        }
    });
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([
            header::CONTENT_TYPE,
            header::AUTHORIZATION,
            HeaderName::from_static("x-cozo-auth"),
        ]);
    let cors = if skip_auth {
        cors.allow_origin(server_origin.parse::<HeaderValue>().unwrap())
    } else {
        cors.allow_origin(Any)
    };

    let app = Router::new()
        .route("/text-query", post(text_query))
        .route("/memory/:op", post(memory_op))
        .route("/tenant/:op", post(tenant_op))
        .route("/oauth/token", post(oauth_token))
        .route("/mcp", post(mcp_post).get(mcp_not_allowed).delete(mcp_not_allowed))
        .route("/flow/:op", post(flow_op))
        .route("/doc/:op", post(doc_op))
        .route("/ontology/:op", post(ontology_op))
        .route("/pg/:op", post(pg_op))
        .route("/pg/debezium/:source", post(pg_debezium))
        .route("/skill/:op", post(skill_op))
        .route("/skills/registry", get(skill_registry_list))
        .route("/skills/registry/:name", get(skill_registry_get))
        .route("/export/:relations", get(export_relations))
        .route("/import", put(import_relations))
        .route("/backup", post(backup))
        .route("/import-from-backup", post(import_from_backup))
        .route("/changes/:relation", get(observe_changes))
        .route("/subscribe", get(subscribe_query))
        .route("/rules/:name", get(register_rule))
        .route(
            "/rule-result/:id",
            post(post_rule_result).delete(post_rule_err),
        ) // +keep alive
        .route("/transact", post(start_transact))
        .route("/transact/:id", post(transact_query).put(finish_query))
        .with_state(state)
        .layer(AsyncRequireAuthorizationLayer::new(auth_obj))
        .fallback(not_found)
        .route("/", get(root))
        .route(
            "/.well-known/oauth-authorization-server",
            get({
                let t = tenancy.clone();
                let o = server_origin.clone();
                move || {
                    let t = t.clone();
                    let o = o.clone();
                    async move {
                        match t.as_ref().filter(|c| c.signing_secret_env.is_some()) {
                            Some(c) => (StatusCode::OK, Json(c.as_metadata(&o))),
                            None => (StatusCode::NOT_FOUND, Json(json!({"error": "token exchange is off"}))),
                        }
                    }
                }
            }),
        )
        .route(
            "/.well-known/oauth-protected-resource",
            get(move || {
                let m = metadata.clone();
                async move {
                    match m.as_ref() {
                        Some(v) => (StatusCode::OK, Json(v.clone())),
                        None => (StatusCode::NOT_FOUND, Json(json!({"error": "principal mode is off"}))),
                    }
                }
            }),
        )
        .layer(cors)
        .layer(CompressionLayer::new())
        .layer(DefaultBodyLimit::disable());

    let addr = if Ipv6Addr::from_str(&args.bind).is_ok() {
        SocketAddr::from_str(&format!("[{}]:{}", args.bind, args.port)).unwrap()
    } else {
        SocketAddr::from_str(&format!("{}:{}", args.bind, args.port)).unwrap()
    };

    if !bind_is_loopback {
        warn!("{}", include_str!("./security.txt"));
    }
    if skip_auth {
        warn!("HTTP authentication is disabled by --insecure-no-auth");
    } else {
        info!("The auth token is in the file: {conf_path}");
    }

    info!(
        "Starting Cozo ({}-backed) API at http://{}",
        args.engine, addr
    );

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

#[derive(serde_derive::Deserialize)]
struct StartTransactPayload {
    write: bool,
}

fn acquire_transaction(
    st: &DbState,
    id: u32,
    auth: &AuthContext,
) -> Result<(Arc<MultiTransaction>, TransactionLease), (StatusCode, Json<serde_json::Value>)> {
    let mut txs = st.txs.lock().unwrap();
    let stored = txs
        .get_mut(&id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, json!({"ok": false}).into()))?;
    if stored.owner != auth.principal {
        return Err((StatusCode::FORBIDDEN, json!({"ok": false}).into()));
    }
    if stored.write && matches!(auth.mutability, ScriptMutability::Immutable) {
        return Err((
            StatusCode::FORBIDDEN,
            json!({"ok": false, "message": "write transaction requires a mutable token"}).into(),
        ));
    }
    if stored.in_flight != 0 {
        return Err((
            StatusCode::CONFLICT,
            json!({"ok": false, "message": "transaction already has an operation in progress"})
                .into(),
        ));
    }
    stored.in_flight += 1;
    let tx = stored.tx.clone();
    drop(txs);
    Ok((
        tx,
        TransactionLease {
            txs: st.txs.clone(),
            id,
        },
    ))
}

async fn start_transact(
    Extension(auth): Extension<AuthContext>,
    State(st): State<DbState>,
    Query(payload): Query<StartTransactPayload>,
) -> (StatusCode, Json<serde_json::Value>) {
    if payload.write && matches!(auth.mutability, ScriptMutability::Immutable) {
        return (StatusCode::FORBIDDEN, json!({"ok": false}).into());
    }
    let tx = st.db.multi_transaction(payload.write);
    let id = st.tx_counter.fetch_add(1, Ordering::SeqCst);
    st.txs.lock().unwrap().insert(
        id,
        StoredTransaction {
            tx: Arc::new(tx),
            owner: auth.principal,
            write: payload.write,
            expires_at: Instant::now() + TRANSACTION_IDLE_TIMEOUT,
            in_flight: 0,
        },
    );
    (StatusCode::OK, json!({"ok": true, "id": id}).into())
}

async fn transact_query(
    Extension(auth): Extension<AuthContext>,
    State(st): State<DbState>,
    Path(id): Path<u32>,
    Json(payload): Json<QueryPayload>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (tx, lease) = match acquire_transaction(&st, id, &auth) {
        Ok(acquired) => acquired,
        Err(response) => return response,
    };
    let src = payload.script.clone();
    let result = spawn_blocking(move || {
        let _lease = lease;
        let params = payload
            .params
            .into_iter()
            .map(|(k, v)| (k, DataValue::from(v)))
            .collect();
        let query = payload.script;
        tx.run_script(&query, params)
    })
    .await;
    match result {
        Ok(Ok(res)) => (StatusCode::OK, res.into_json().into()),
        Ok(Err(err)) => (
            StatusCode::BAD_REQUEST,
            format_error_as_json(err, Some(&src)).into(),
        ),
        Err(err) => internal_error(err),
    }
}

#[derive(serde_derive::Deserialize)]
struct FinishTransactPayload {
    abort: bool,
}

async fn finish_query(
    Extension(auth): Extension<AuthContext>,
    State(st): State<DbState>,
    Path(id): Path<u32>,
    Json(payload): Json<FinishTransactPayload>,
) -> (StatusCode, Json<serde_json::Value>) {
    let tx = match st.txs.lock().unwrap().entry(id) {
        std::collections::btree_map::Entry::Vacant(_) => {
            return (StatusCode::NOT_FOUND, json!({"ok": false}).into())
        }
        std::collections::btree_map::Entry::Occupied(entry)
            if entry.get().owner != auth.principal =>
        {
            return (StatusCode::FORBIDDEN, json!({"ok": false}).into())
        }
        std::collections::btree_map::Entry::Occupied(entry)
            if entry.get().write
                && matches!(auth.mutability, ScriptMutability::Immutable)
                && !payload.abort =>
        {
            return (
                StatusCode::FORBIDDEN,
                json!({"ok": false, "message": "write transaction requires a mutable token"})
                    .into(),
            )
        }
        std::collections::btree_map::Entry::Occupied(entry) if entry.get().in_flight != 0 => {
            return (
                StatusCode::CONFLICT,
                json!({"ok": false, "message": "transaction has an operation in progress"}).into(),
            )
        }
        std::collections::btree_map::Entry::Occupied(entry) => entry.remove().tx,
    };
    let res = if payload.abort {
        tx.abort()
    } else {
        tx.commit()
    };
    match res {
        Ok(_) => (StatusCode::OK, json!({"ok": true}).into()),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            json!({"ok": false, "message": err.to_string()}).into(),
        ),
    }
}

#[derive(serde_derive::Deserialize)]
struct QueryPayload {
    script: String,
    params: BTreeMap<String, serde_json::Value>,
    immutable: Option<bool>,
    /// Per-call wall-clock budget in seconds (kleos fork, query budget).
    /// Optional; combined via `min` with any in-script `:timeout` and the
    /// server's `--default-query-timeout`. Ignored on the `/transact/:id` path.
    timeout: Option<f64>,
    /// Per-call memory budget in estimated bytes (kleos fork, query memory
    /// budget). Optional; combined via `min` with any in-script `:mem_limit`
    /// and the server's `--default-query-mem-limit`.
    mem_limit: Option<usize>,
}

async fn text_query(
    Extension(mutability): Extension<ScriptMutability>,
    State(st): State<DbState>,
    Json(payload): Json<QueryPayload>,
) -> (StatusCode, Json<serde_json::Value>) {
    let params = payload
        .params
        .into_iter()
        .map(|(k, v)| (k, DataValue::from(v)))
        .collect();
    let immutable = match mutability {
        ScriptMutability::Mutable => payload.immutable.unwrap_or(false),
        ScriptMutability::Immutable => true,
    };
    let options = ScriptRunOptions {
        timeout: payload.timeout,
        mem_limit: payload.mem_limit,
    };
    let result = spawn_blocking(move || {
        st.db.run_script_fold_err_with_options(
            &payload.script,
            params,
            if immutable {
                ScriptMutability::Immutable
            } else {
                ScriptMutability::Mutable
            },
            options,
        )
    })
    .await;
    match result {
        Ok(res) => wrap_json(res),
        Err(err) => internal_error(err),
    }
}

/// Agent memory API (kleos fork; docs/specs/memory-api.md): `POST
/// /memory/{init|remember|recall|forget|reflect|link|set|flush}`, and the
/// branch ops `fork|rewind|merge|diff|discard|branches`
/// (docs/specs/memory-branches.md), with the op's options
/// as a JSON object (`text` for remember, `query` for recall, `id`/`ids` for
/// forget). Answers `{"ok": true, "result": ...}`. Only `recall`,
/// `reflect`, `diff` and `branches` are allowed on a read-only (immutable)
/// principal.
async fn memory_op(
    Extension(mutability): Extension<ScriptMutability>,
    Extension(auth): Extension<AuthContext>,
    State(st): State<DbState>,
    Path(op): Path<String>,
    Json(args): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    // principal mode: the op runs as the authenticated principal
    if let Some(ctx) = auth.tenant.clone() {
        let reads = ["recall", "context", "reflect", "diff", "scopes"];
        if matches!(mutability, ScriptMutability::Immutable) && !reads.contains(&op.as_str()) {
            return (StatusCode::FORBIDDEN, json!({"ok": false, "message": format!("memory {op} needs write access")}).into());
        }
        let result = spawn_blocking(move || st.db.memory_call_as(&ctx, &op, args)).await;
        return match result {
            Ok(Ok(v)) => (StatusCode::OK, json!({"ok": true, "result": v}).into()),
            Ok(Err(err)) => (StatusCode::BAD_REQUEST, json!({"ok": false, "message": err.to_string()}).into()),
            Err(err) => internal_error(err),
        };
    }
    if matches!(mutability, ScriptMutability::Immutable) && !["recall", "reflect", "diff", "branches"].contains(&op.as_str()) {
        return (
            StatusCode::FORBIDDEN,
            json!({"ok": false, "message": format!("memory {op} needs write access")}).into(),
        );
    }
    let mut args = args;
    if matches!(mutability, ScriptMutability::Immutable) && op == "recall" {
        // a read-only principal's recalls are not counted as accesses
        if let serde_json::Value::Object(m) = &mut args {
            m.insert("record_access".to_string(), serde_json::Value::Bool(false));
        }
    }
    let result = spawn_blocking(move || st.db.memory_call(&op, args)).await;
    match result {
        Ok(Ok(v)) => (StatusCode::OK, json!({"ok": true, "result": v}).into()),
        Ok(Err(err)) => (
            StatusCode::BAD_REQUEST,
            json!({"ok": false, "message": err.to_string()}).into(),
        ),
        Err(err) => internal_error(err),
    }
}

/// `GET`/`DELETE /mcp`: protocol 2026-07-28 has no standalone stream and no
/// sessions.
async fn mcp_not_allowed() -> StatusCode {
    StatusCode::METHOD_NOT_ALLOWED
}

fn principal_name(p: &Principal) -> String {
    match p {
        Principal::Local => "local".into(),
        Principal::Admin => "admin".into(),
        Principal::Bearer(h) => h.iter().map(|b| format!("{b:02x}")).collect(),
        Principal::Tenant(id) => format!("tenant:{id}"),
    }
}

/// RFC 8693 token exchange (docs/specs/tenancy.md): `POST /oauth/token`
/// with a form body; no other credential (the subject and actor tokens are
/// the credentials).
async fn oauth_token(
    State(st): State<DbState>,
    headers: axum::http::HeaderMap,
    axum::extract::Form(form): axum::extract::Form<std::collections::HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let Some(cfg) = st.tenancy.clone() else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "invalid_request", "error_description": "principal mode is off"})));
    };
    let info = kleos_bin::auth::RequestInfo { method: "POST".into(), url: format!("{}/oauth/token", st.origin.trim_end_matches('/')) };
    let db = st.db.clone();
    let res = spawn_blocking(move || {
        let h = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).map(|x| x.to_string());
        kleos_bin::auth::token_exchange(&cfg, &db, &form, &h, &info)
    })
    .await;
    match res {
        Ok(Ok(v)) => (StatusCode::OK, Json(v)),
        Ok(Err(e)) => (
            StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_REQUEST),
            Json(json!({"error": e.error, "error_description": e.description})),
        ),
        Err(e) => internal_error(e),
    }
}

/// Tenancy admin ops (docs/specs/tenancy.md): `POST /tenant/:op`. The admin
/// secret may run any; a principal must be an admin of the workspace's org
/// (and its own credential must allow `admin`).
async fn tenant_op(
    Extension(auth): Extension<AuthContext>,
    State(st): State<DbState>,
    Path(op): Path<String>,
    Json(args): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let result = spawn_blocking(move || -> miette::Result<serde_json::Value> {
        match (&auth.principal, auth.tenant.as_deref()) {
            (Principal::Admin | Principal::Local, _) => st.db.tenant_call(&op, args),
            (_, Some(ctx)) => {
                let who = ctx.human.clone().ok_or_else(|| miette!("tenant ops need a human principal"))?;
                if ctx.token.as_ref().and_then(|t| t.actions.as_ref()).is_some_and(|a| !a.iter().any(|x| x == "admin")) {
                    miette::bail!("this credential does not allow admin ops");
                }
                // the org the op touches, from every object it names; an op
                // that names none (e.g. `list`) needs the admin secret
                let org = st.db.tenant_target_org(&args)?.ok_or_else(|| miette!("tenant {op}: name the org, workspace or principal it acts on"))?;
                if op == "list" || op == "enable" {
                    miette::bail!("tenant {op} needs the admin secret");
                }
                if !st.db.is_org_admin(&who, &org)? {
                    miette::bail!("'{who}' is not an admin of org '{org}'");
                }
                st.db.tenant_call(&op, args)
            }
            _ => miette::bail!("tenant ops need the admin secret or a tenancy principal"),
        }
    })
    .await;
    match result {
        Ok(Ok(v)) => (StatusCode::OK, json!({"ok": true, "result": v}).into()),
        Ok(Err(err)) => (StatusCode::FORBIDDEN, json!({"ok": false, "message": err.to_string()}).into()),
        Err(err) => internal_error(err),
    }
}

fn mcp_json(status: u16, body: serde_json::Value) -> Response<Body> {
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::OK))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Whether an `Origin` may call `/mcp`: the server's own origin, or any
/// loopback origin when the server itself is on loopback.
fn mcp_origin_ok(origin: &str, server_origin: &str) -> bool {
    let loopback = |o: &str| {
        let rest = o.split_once("://").map(|x| x.1).unwrap_or(o);
        let host = if rest.starts_with('[') { rest.split(']').next().unwrap_or("").trim_start_matches('[') } else { rest.split(':').next().unwrap_or("") };
        matches!(host, "localhost" | "127.0.0.1" | "::1")
    };
    origin == server_origin || (loopback(server_origin) && loopback(origin))
}

/// `POST /mcp`: one JSON-RPC message per request (Streamable HTTP, protocol
/// 2026-07-28; docs/specs/mcp-cli-skills.md). Validates `Origin` and the
/// mirrored `MCP-Protocol-Version` / `Mcp-Method` / `Mcp-Name` headers, then
/// hands the message to the stateless MCP core. No session ids are minted;
/// `Mcp-Session-Id` and `Last-Event-ID` are ignored.
async fn mcp_post(
    Extension(auth): Extension<AuthContext>,
    State(st): State<DbState>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response<Body> {
    use kleos_bin::mcp::{decode_header_value, Conn, RpcError, HEADER_MISMATCH, LEGACY, MODERN, PARSE_ERROR};
    let Some(server) = st.mcp.clone() else {
        return Response::builder().status(StatusCode::NOT_FOUND).body(Body::empty()).unwrap();
    };
    if let Some(origin) = headers.get(header::ORIGIN).and_then(|o| o.to_str().ok()) {
        if !mcp_origin_ok(origin, &st.origin) {
            return mcp_json(403, RpcError::new(-32600, "Origin not allowed").to_response(None));
        }
    }
    let msg: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return mcp_json(400, RpcError::new(PARSE_ERROR, format!("Parse error: {e}")).to_response(None)),
    };
    let id = msg.get("id").cloned();
    let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).map(|s| s.to_string());
    let mismatch = |why: String| mcp_json(400, RpcError::new(HEADER_MISMATCH, why).to_response(id.as_ref()));
    let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or_default().to_string();
    let body_version = msg.pointer("/params/_meta/io.modelcontextprotocol~1protocolVersion").and_then(|v| v.as_str()).map(|s| s.to_string());
    let header_version = header("mcp-protocol-version");
    let mut conn = Conn {
        principal: principal_name(&auth.principal),
        read_only: matches!(auth.mutability, ScriptMutability::Immutable),
        legacy: None,
        as_principal: auth.tenant.as_deref().cloned(),
    };
    if let Some(bv) = &body_version {
        // modern request: the mirrored headers are required and must match
        match &header_version {
            None => return mismatch("missing MCP-Protocol-Version header".into()),
            Some(hv) if hv != bv => return mismatch(format!("MCP-Protocol-Version header '{hv}' does not match body value '{bv}'")),
            _ => {}
        }
        match header("mcp-method") {
            None => return mismatch("missing Mcp-Method header".into()),
            Some(hm) if hm != method => return mismatch(format!("Mcp-Method header '{hm}' does not match body value '{method}'")),
            _ => {}
        }
        let name_field = match method.as_str() {
            "tools/call" | "prompts/get" => msg.pointer("/params/name"),
            "resources/read" => msg.pointer("/params/uri"),
            _ => None,
        };
        if let Some(expected) = name_field.and_then(|v| v.as_str()) {
            match header("mcp-name").map(|h| decode_header_value(&h)) {
                None => return mismatch("missing Mcp-Name header".into()),
                Some(None) => return mismatch("malformed Mcp-Name header".into()),
                Some(Some(hn)) if hn != expected => return mismatch(format!("Mcp-Name header '{hn}' does not match body value '{expected}'")),
                _ => {}
            }
        }
    } else if method != "initialize" {
        // an earlier-revision client: its version comes from the header
        // (2025-06-18 and later); without one, 2025-03-26
        match header_version.as_deref() {
            Some(v) if v == MODERN => {} // modern header, but no _meta: the core rejects it
            Some(v) if LEGACY.contains(&v) => conn.legacy = Some(v.to_string()),
            Some(_) => {}
            None => conn.legacy = Some("2025-03-26".to_string()),
        }
    }
    let result = spawn_blocking(move || server.handle(&msg, &mut conn)).await;
    match result {
        Ok(Some((resp, status))) => mcp_json(status, resp),
        Ok(None) => Response::builder().status(StatusCode::ACCEPTED).body(Body::empty()).unwrap(),
        Err(err) => mcp_json(500, RpcError::new(-32603, format!("internal error: {err}")).to_response(None)),
    }
}

/// Durable workflows (kleos fork; docs/specs/durable-workflows.md): `POST
/// /flow/{start|signal|cancel|status|list|history|claim|step|...}` with the
/// op's options as a JSON object. Answers `{"ok": true, "result": ...}`. A
/// read-only principal may only read (`status`, `list`, `history`, `stats`,
/// `schedules`, `next_due`).
async fn flow_op(
    Extension(mutability): Extension<ScriptMutability>,
    State(st): State<DbState>,
    Path(op): Path<String>,
    Json(args): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let reads = ["status", "list", "history", "stats", "schedules", "next_due"];
    if matches!(mutability, ScriptMutability::Immutable) && !reads.contains(&op.as_str()) {
        return (
            StatusCode::FORBIDDEN,
            json!({"ok": false, "message": format!("flow {op} needs write access")}).into(),
        );
    }
    let result = spawn_blocking(move || st.db.flow_call(&op, args)).await;
    match result {
        Ok(Ok(v)) => (StatusCode::OK, json!({"ok": true, "result": v}).into()),
        Ok(Err(err)) => (
            StatusCode::BAD_REQUEST,
            json!({"ok": false, "message": err.to_string()}).into(),
        ),
        Err(err) => internal_error(err),
    }
}

/// Ontologies (kleos fork; docs/specs/ontology.md): `POST
/// /ontology/{define|import|export|get|list|activate|assert|facts|infer|...}`
/// with the op's options as a JSON object. A read-only principal may only
/// read (`get`, `list`, `export`, `facts`, `infer`, `validate`, `type`).
async fn ontology_op(
    Extension(mutability): Extension<ScriptMutability>,
    State(st): State<DbState>,
    Path(op): Path<String>,
    Json(args): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let reads = ["get", "list", "export", "facts", "infer", "validate", "type", "gate"];
    if matches!(mutability, ScriptMutability::Immutable) && !reads.contains(&op.as_str()) {
        return (
            StatusCode::FORBIDDEN,
            json!({"ok": false, "message": format!("ontology {op} needs write access")}).into(),
        );
    }
    let result = spawn_blocking(move || st.db.ontology_call(&op, args)).await;
    match result {
        Ok(Ok(v)) => (StatusCode::OK, json!({"ok": true, "result": v}).into()),
        Ok(Err(err)) => (
            StatusCode::BAD_REQUEST,
            json!({"ok": false, "message": err.to_string()}).into(),
        ),
        Err(err) => internal_error(err),
    }
}

/// Start the `--pg-cdc` connectors (and the `pg` flow action with `flow-pg`).
#[cfg(feature = "pg-cdc")]
fn start_pg(configs: &[String], db: &DbInstance) {
    #[cfg(feature = "flow-pg")]
    if let Err(e) = kleos_pg::writeback::register_flow_action() {
        panic!("flow action pg: {e:?}");
    }
    for path in configs {
        let cfg = kleos_pg::PgConfig::load(path).unwrap_or_else(|e| panic!("--pg-cdc {path}: {e:?}"));
        eprintln!("pg connector '{}' (slot {}, publication {}) into namespace '{}'", cfg.source, cfg.slot(), cfg.publication(), cfg.ns);
        kleos_pg::spawn(cfg, db.clone());
    }
}

#[cfg(not(feature = "pg-cdc"))]
fn start_pg(configs: &[String], _db: &DbInstance) {
    if !configs.is_empty() {
        panic!("--pg-cdc needs kleos built with the `pg-cdc` feature");
    }
}

/// The Postgres connector (kleos fork; docs/specs/pg-connector.md): `POST
/// /pg/{status|init|resync|drop_slot|map|approve|query|writeback}` with
/// `{source, …}`. A read-only principal may only read (`status`, `map`,
/// `query`).
async fn pg_op(
    Extension(mutability): Extension<ScriptMutability>,
    State(st): State<DbState>,
    Path(op): Path<String>,
    Json(args): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let reads = ["status", "map", "query"];
    if matches!(mutability, ScriptMutability::Immutable) && !reads.contains(&op.as_str()) {
        return (
            StatusCode::FORBIDDEN,
            json!({"ok": false, "message": format!("pg {op} needs write access")}).into(),
        );
    }
    if op == "map" && args.get("approve").is_some() && matches!(mutability, ScriptMutability::Immutable) {
        return (StatusCode::FORBIDDEN, json!({"ok": false, "message": "pg approve needs write access"}).into());
    }
    let backend = kleos_bin::backend::Backend::Embedded(st.db.clone());
    let result = spawn_blocking(move || backend.call(kleos_bin::backend::Area::Pg, &op, args)).await;
    match result {
        Ok(Ok(v)) => (StatusCode::OK, json!({"ok": true, "result": v}).into()),
        Ok(Err(err)) => (StatusCode::BAD_REQUEST, json!({"ok": false, "message": err}).into()),
        Err(err) => internal_error(err),
    }
}

/// Debezium Server's HTTP sink target (kleos fork; docs/specs/pg-connector.md,
/// "Debezium adapter"): `POST /pg/debezium/<source>` with one change event
/// or a list. Send the auth token with
/// `debezium.sink.http.headers.x-cozo-auth=<token>`.
async fn pg_debezium(
    Extension(mutability): Extension<ScriptMutability>,
    State(st): State<DbState>,
    Path(source): Path<String>,
    Json(events): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    if matches!(mutability, ScriptMutability::Immutable) {
        return (StatusCode::FORBIDDEN, json!({"ok": false, "message": "debezium ingest needs write access"}).into());
    }
    let backend = kleos_bin::backend::Backend::Embedded(st.db.clone());
    let result = spawn_blocking(move || backend.call(kleos_bin::backend::Area::Pg, "debezium", json!({"source": source, "events": events}))).await;
    match result {
        Ok(Ok(v)) => (StatusCode::OK, json!({"ok": true, "result": v}).into()),
        // a failed event must be retried by the sender: 5xx
        Ok(Err(err)) => (StatusCode::SERVICE_UNAVAILABLE, json!({"ok": false, "message": err}).into()),
        Err(err) => internal_error(err),
    }
}

/// Documents (kleos fork; docs/specs/documents.md): `POST /doc/{get|list|delete}`
/// with `{ns, doc, mode}`. Documents themselves go to `POST /memory/remember`
/// with `attachments: [{data_b64, mime, name}]`. Needs the `docs` feature.
async fn doc_op(
    Extension(mutability): Extension<ScriptMutability>,
    State(st): State<DbState>,
    Path(op): Path<String>,
    Json(args): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    if matches!(mutability, ScriptMutability::Immutable) && !["get", "list"].contains(&op.as_str()) {
        return (
            StatusCode::FORBIDDEN,
            json!({"ok": false, "message": format!("doc {op} needs write access")}).into(),
        );
    }
    #[cfg(feature = "docs")]
    let result = spawn_blocking(move || st.db.doc_call(&op, args)).await;
    #[cfg(not(feature = "docs"))]
    let result: Result<miette::Result<serde_json::Value>, tokio::task::JoinError> = {
        let _ = (st, op, args);
        Ok(Err(miette::miette!("documents need kleos-bin built with the `docs` feature")))
    };
    match result {
        Ok(Ok(v)) => (StatusCode::OK, json!({"ok": true, "result": v}).into()),
        Ok(Err(err)) => (
            StatusCode::BAD_REQUEST,
            json!({"ok": false, "message": err.to_string()}).into(),
        ),
        Err(err) => internal_error(err),
    }
}

/// The skill library (kleos fork; docs/specs/skills.md): `POST /skill/:op`
/// with the op's JSON options, as `::skill <op>`. Read-only principals may
/// call only the read ops.
async fn skill_op(
    Extension(mutability): Extension<ScriptMutability>,
    State(st): State<DbState>,
    Path(op): Path<String>,
    Json(args): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    const READS: [&str; 18] = [
        "get", "list", "history", "as_of", "find", "entry", "entries", "read", "dir", "stats", "traces", "evals", "scan", "proposals",
        "promotions", "keys", "verify", "registry",
    ];
    let mut args = args;
    if matches!(mutability, ScriptMutability::Immutable) {
        if !READS.contains(&op.as_str()) {
            return (StatusCode::FORBIDDEN, json!({"ok": false, "message": format!("skill {op} needs write access")}).into());
        }
        if op == "find" {
            if let serde_json::Value::Object(m) = &mut args {
                m.insert("log".into(), json!(false));
            }
        }
    }
    let result = spawn_blocking(move || st.db.skill_call(&op, args)).await;
    match result {
        Ok(Ok(v)) => (StatusCode::OK, json!({"ok": true, "result": v}).into()),
        Ok(Err(err)) => (StatusCode::BAD_REQUEST, json!({"ok": false, "message": err.to_string()}).into()),
        Err(err) => internal_error(err),
    }
}

/// The skill registry of this store: `GET /skills/registry?query=` lists the
/// published packages, `GET /skills/registry/:name` returns one signed
/// package (what `::skill install {url, name}` fetches from another store).
async fn skill_registry_list(
    State(st): State<DbState>,
    Query(q): Query<BTreeMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let args = json!({"query": q.get("query")});
    match spawn_blocking(move || st.db.skill_call("registry", args)).await {
        Ok(Ok(v)) => (StatusCode::OK, json!({"ok": true, "result": v}).into()),
        Ok(Err(err)) => (StatusCode::BAD_REQUEST, json!({"ok": false, "message": err.to_string()}).into()),
        Err(err) => internal_error(err),
    }
}

async fn skill_registry_get(State(st): State<DbState>, Path(name): Path<String>) -> (StatusCode, Json<serde_json::Value>) {
    match spawn_blocking(move || st.db.skill_call("registry", json!({"name": name}))).await {
        Ok(Ok(v)) => (StatusCode::OK, json!({"ok": true, "result": v}).into()),
        Ok(Err(err)) => (StatusCode::NOT_FOUND, json!({"ok": false, "message": err.to_string()}).into()),
        Err(err) => internal_error(err),
    }
}

async fn export_relations(
    State(st): State<DbState>,
    Path(relations): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let relations = relations
        .split(',')
        .filter_map(|t| {
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        })
        .collect_vec();
    let result = spawn_blocking(move || st.db.export_relations(relations.iter())).await;
    match result {
        Ok(Ok(s)) => {
            let s: serde_json::Map<_, _> = s.into_iter().map(|(k, v)| (k, v.into_json())).collect();
            let ret = json!({"ok": true, "data": s});
            (StatusCode::OK, ret.into())
        }
        Ok(Err(err)) => {
            let ret = json!({"ok": false, "message": err.to_string()});
            (StatusCode::BAD_REQUEST, ret.into())
        }
        Err(err) => internal_error(err),
    }
}

async fn import_relations(
    Extension(mutability): Extension<ScriptMutability>,
    State(st): State<DbState>,
    Json(payload): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(response) = require_mutable(mutability) {
        return response;
    }

    let payload = match payload.as_object() {
        None => {
            return (
                StatusCode::BAD_REQUEST,
                json!({"ok": false, "message": "payload must be a JSON object"}).into(),
            );
        }
        Some(pl) => {
            let mut ret = BTreeMap::new();
            for (k, v) in pl {
                let nr = match NamedRows::from_json(v) {
                    Ok(p) => p,
                    Err(err) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            json!({"ok": false, "message": err.to_string()}).into(),
                        );
                    }
                };
                ret.insert(k.to_string(), nr);
            }
            ret
        }
    };

    let result = spawn_blocking(move || st.db.import_relations(payload)).await;
    match result {
        Ok(Ok(_)) => (StatusCode::OK, json!({"ok": true}).into()),
        Ok(Err(err)) => {
            let ret = json!({"ok": false, "message": err.to_string()});
            (StatusCode::BAD_REQUEST, ret.into())
        }
        Err(err) => internal_error(err),
    }
}

#[derive(serde_derive::Deserialize)]
struct BackupPayload {
    path: String,
}

fn resolve_backup_path(backup_dir: &FsPath, requested: &str) -> miette::Result<PathBuf> {
    let requested = FsPath::new(requested);
    if requested.as_os_str().is_empty() || requested.is_absolute() {
        return Err(miette!(
            "backup path must be relative to the HTTP backup directory"
        ));
    }

    let candidate = backup_dir.join(requested);
    let parent = candidate
        .parent()
        .ok_or_else(|| miette!("backup path has no parent directory"))?;
    let resolved_parent =
        std::fs::canonicalize(parent).map_err(|_| miette!("backup path parent does not exist"))?;
    if !resolved_parent.starts_with(backup_dir) {
        return Err(miette!("backup path escapes the HTTP backup directory"));
    }

    // Canonicalize an existing file as well, so a symlink cannot point outside
    // the configured directory. New files inherit the checked parent.
    match std::fs::symlink_metadata(&candidate) {
        Ok(_) => {
            let resolved = std::fs::canonicalize(&candidate)
                .map_err(|_| miette!("cannot resolve backup path"))?;
            if !resolved.starts_with(backup_dir) {
                return Err(miette!("backup path escapes the HTTP backup directory"));
            }
            Ok(resolved)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Ok(resolved_parent.join(candidate.file_name().unwrap()))
        }
        Err(_) => Err(miette!("cannot inspect backup path")),
    }
}

#[cfg(test)]
mod backup_path_tests {
    use super::resolve_backup_path;
    use std::fs;

    #[test]
    fn confines_http_backup_paths_to_configured_directory() {
        let root = std::env::temp_dir().join(format!("cozo-backup-path-{}", std::process::id()));
        let backups = root.join("backups");
        fs::create_dir_all(backups.join("nested")).unwrap();
        let backups = fs::canonicalize(backups).unwrap();

        assert_eq!(
            resolve_backup_path(&backups, "nested/safe.db").unwrap(),
            backups.join("nested/safe.db")
        );
        assert!(resolve_backup_path(&backups, "/tmp/outside.db").is_err());
        assert!(resolve_backup_path(&backups, "../outside.db").is_err());

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.join("missing-outside.db"), backups.join("link.db"))
                .unwrap();
            assert!(resolve_backup_path(&backups, "link.db").is_err());
        }

        fs::remove_dir_all(root).unwrap();
    }
}

async fn backup(
    Extension(mutability): Extension<ScriptMutability>,
    State(st): State<DbState>,
    Json(payload): Json<BackupPayload>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(response) = require_mutable(mutability) {
        return response;
    }

    let path = match resolve_backup_path(&st.backup_dir, &payload.path) {
        Ok(path) => path,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                json!({"ok": false, "message": err.to_string()}).into(),
            );
        }
    };
    let result = spawn_blocking(move || st.db.backup_db(path)).await;

    match result {
        Ok(Ok(())) => {
            let ret = json!({"ok": true});
            (StatusCode::OK, ret.into())
        }
        Ok(Err(err)) => {
            let ret = json!({"ok": false, "message": err.to_string()});
            (StatusCode::BAD_REQUEST, ret.into())
        }
        Err(err) => internal_error(err),
    }
}

#[derive(serde_derive::Deserialize)]
struct BackupImportPayload {
    path: String,
    relations: Vec<String>,
}

async fn import_from_backup(
    Extension(mutability): Extension<ScriptMutability>,
    State(st): State<DbState>,
    Json(payload): Json<BackupImportPayload>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(response) = require_mutable(mutability) {
        return response;
    }

    let path = match resolve_backup_path(&st.backup_dir, &payload.path) {
        Ok(path) => path,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                json!({"ok": false, "message": err.to_string()}).into(),
            );
        }
    };
    let result = spawn_blocking(move || st.db.import_from_backup(path, &payload.relations)).await;

    match result {
        Ok(Ok(())) => {
            let ret = json!({"ok": true});
            (StatusCode::OK, ret.into())
        }
        Ok(Err(err)) => {
            let ret = json!({"ok": false, "message": err.to_string()});
            (StatusCode::BAD_REQUEST, ret.into())
        }
        Err(err) => internal_error(err),
    }
}

#[derive(serde_derive::Deserialize)]
struct RuleRegisterOptions {
    arity: usize,
}

async fn post_rule_result(
    State(st): State<DbState>,
    Path(id): Path<u32>,
    Json(res): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let res = match NamedRows::from_json(&res) {
        Ok(res) => res,
        Err(err) => {
            if let Some(ch) = st.rule_senders.lock().unwrap().remove(&id) {
                let _ = ch.send(Err(miette!("downstream posted malformed result")));
            }
            return (
                StatusCode::BAD_REQUEST,
                json!({"ok": false, "message": err.to_string()}).into(),
            );
        }
    };
    if let Some(ch) = st.rule_senders.lock().unwrap().remove(&id) {
        match ch.send(Ok(res)) {
            Ok(_) => (StatusCode::OK, json!({"ok": true}).into()),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"ok": false, "message": err.to_string()}).into(),
            ),
        }
    } else {
        (StatusCode::NOT_FOUND, json!({"ok": false}).into())
    }
}

async fn post_rule_err(
    State(st): State<DbState>,
    Path(id): Path<u32>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Some(ch) = st.rule_senders.lock().unwrap().remove(&id) {
        match ch.send(Err(miette!("downstream cancelled computation"))) {
            Ok(_) => (StatusCode::OK, json!({"ok": true}).into()),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"ok": false, "message": err.to_string()}).into(),
            ),
        }
    } else {
        (StatusCode::NOT_FOUND, json!({"ok": false}).into())
    }
}

async fn register_rule(
    State(st): State<DbState>,
    Path(name): Path<String>,
    Query(rule_opts): Query<RuleRegisterOptions>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (rule, task_receiver) = SimpleFixedRule::rule_with_channel(rule_opts.arity);
    let (down_sender, mut down_receiver) = tokio::sync::mpsc::channel(1);
    let mut errored = None;

    if let Err(err) = st.db.register_fixed_rule(name.clone(), rule) {
        errored = Some(err);
    } else {
        let rule_senders = st.rule_senders.clone();
        let rule_counter = st.rule_counter.clone();

        rayon::spawn(move || {
            for (inputs, options, sender) in task_receiver {
                let id = rule_counter.fetch_add(1, Ordering::AcqRel);
                let inputs: serde_json::Value =
                    inputs.into_iter().map(|ip| ip.into_json()).collect();
                let options: serde_json::Value = options
                    .into_iter()
                    .map(|(k, v)| (k, serde_json::Value::from(v)))
                    .collect();
                if down_sender.blocking_send((id, inputs, options)).is_err() {
                    let _ = sender.send(Err(miette!("cannot send request to downstream")));
                } else {
                    rule_senders.lock().unwrap().insert(id, sender);
                }
            }
        });
    }

    struct Guard {
        name: String,
        db: DbInstance,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            info!("dropping rules SSE {}", self.name);
            let _ = self.db.unregister_fixed_rule(&self.name);
        }
    }

    let stream = async_stream::stream! {
        if let Some(err) = errored {
            let item = json!({"type": "register-error", "error": err.to_string()});
            yield Ok(Event::default().json_data(item).unwrap());
        } else {
            info!("starting rule SSE {}", name);
            let _guard = Guard {db: st.db, name};
            while let Some((id, inputs, options)) = down_receiver.recv().await {
                let item = json!({"type": "request", "id": id, "inputs": inputs, "options": options});
                yield Ok(Event::default().json_data(item).unwrap());
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn observe_changes(
    State(st): State<DbState>,
    Path(relation): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (id, recv) = st.db.register_callback(&relation, None);
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    struct Guard {
        id: u32,
        db: DbInstance,
        relation: String,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            info!("dropping changes SSE {}: {}", self.relation, self.id);
            self.db.unregister_callback(self.id);
        }
    }

    spawn_blocking(move || {
        for data in recv {
            sender.blocking_send(data).unwrap();
        }
    });
    let stream = async_stream::stream! {
        info!("starting changes SSE {}: {}", relation, id);
        let _guard = Guard {id, db: st.db, relation};
        while let Some((op, new, old)) = receiver.recv().await {
            let item = json!({"op": op.to_string(), "new_rows": new.into_json(), "old_rows": old.into_json()});
            yield Ok(Event::default().json_data(item).unwrap());
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(serde_derive::Deserialize)]
struct SubscribeParams {
    script: String,
    /// JSON object of query parameters, e.g. `{"min": 2}`.
    params: Option<String>,
    /// Per-subscriber queue capacity (events); default 64.
    queue: Option<usize>,
}

/// Live query subscription over server-sent events (UPGRADE_PLAN F1;
/// `docs/specs/live-subscriptions.md`). `GET /subscribe?script=...&params=...`
/// streams `initial`, then `delta` / `resync` / `error` events as JSON. The
/// subscription ends when the client disconnects.
async fn subscribe_query(
    State(st): State<DbState>,
    Query(q): Query<SubscribeParams>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let params: BTreeMap<String, DataValue> = match q.params.as_deref() {
        None | Some("") => BTreeMap::new(),
        Some(p) => match serde_json::from_str::<BTreeMap<String, serde_json::Value>>(p) {
            Ok(m) => m.into_iter().map(|(k, v)| (k, DataValue::from(v))).collect(),
            Err(e) => {
                return wrap_json(json!({"ok": false, "message": format!("bad params: {e}")}))
                    .into_response()
            }
        },
    };
    let opts = SubscribeOptions {
        queue_capacity: q.queue.unwrap_or(SubscribeOptions::default().queue_capacity),
    };
    let db = st.db.clone();
    let script = q.script.clone();
    let sub = match spawn_blocking(move || db.subscribe(&script, params, opts)).await {
        Ok(Ok(sub)) => sub,
        Ok(Err(err)) => {
            return wrap_json(format_error_as_json(err, Some(&q.script))).into_response()
        }
        Err(err) => return internal_error(err).into_response(),
    };
    let id = sub.id();
    let (sender, mut receiver) = tokio::sync::mpsc::channel::<SubscriptionEvent>(16);
    spawn_blocking(move || {
        // Poll so a disconnected client releases its subscription even when
        // no event is coming.
        while !sender.is_closed() {
            if let Some(ev) = sub.recv_timeout(Duration::from_secs(1)) {
                if sender.blocking_send(ev).is_err() {
                    break;
                }
            }
        }
        info!("subscription SSE {id} closed");
    });
    let stream = async_stream::stream! {
        info!("starting subscription SSE {id}");
        while let Some(ev) = receiver.recv().await {
            let (name, item) = match ev {
                SubscriptionEvent::Initial { rows } => ("initial", json!({"rows": (*rows).clone().into_json()})),
                SubscriptionEvent::Delta { added, removed } => ("delta", json!({
                    "added": (*added).clone().into_json(),
                    "removed": (*removed).clone().into_json(),
                })),
                SubscriptionEvent::Resync { rows } => ("resync", json!({"rows": (*rows).clone().into_json()})),
                SubscriptionEvent::Error { message } => ("error", json!({"message": message})),
            };
            yield Ok::<_, Infallible>(Event::default().event(name).json_data(item).unwrap());
        }
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn root() -> Html<&'static str> {
    Html(include_str!("./index.html"))
}

fn internal_error<E>(err: E) -> (StatusCode, Json<serde_json::Value>)
where
    E: std::error::Error,
{
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"ok": false, "message": err.to_string()}).into(),
    )
}

fn wrap_json(json: serde_json::Value) -> (StatusCode, Json<serde_json::Value>) {
    let code = if let Some(serde_json::Value::Bool(true)) = json.get("ok") {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    };
    (code, json.into())
}

pub async fn not_found(uri: axum::http::Uri) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::NOT_FOUND,
        json!({"ok": false, "message": format!("No route {}", uri)}).into(),
    )
}
