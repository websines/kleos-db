/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Connections to Postgres (plain, replication, TLS) and the small runtime
//! that lets synchronous callers (the kleos flow worker, the CLI) run the
//! connector's async code.

use std::future::Future;
use std::str::FromStr;
use std::sync::OnceLock;

use miette::{miette, Result};
use tokio_postgres::config::{ReplicationMode, SslMode};
use tokio_postgres::{Client, Config};

/// The connector's own runtime for synchronous callers.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("kleos-pg")
            .enable_all()
            .build()
            .expect("kleos-pg runtime")
    })
}

/// Run `fut` to completion from synchronous code: spawned on the
/// connector's runtime and awaited through a channel, so it works from any
/// thread that is not one of that runtime's workers (a flow worker thread,
/// a `spawn_blocking` thread, the CLI's main thread).
pub fn block_on<F, T>(fut: F) -> T
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    runtime().spawn(async move {
        let _ = tx.send(fut.await);
    });
    rx.recv().expect("kleos-pg task panicked")
}

pub fn pg_err(e: tokio_postgres::Error) -> miette::Report {
    match e.as_db_error() {
        Some(d) => miette!("postgres: {} ({}){}", d.message(), d.code().code(), d.detail().map(|x| format!(": {x}")).unwrap_or_default()),
        None => miette!("postgres: {e}"),
    }
}

/// Parse a connection string; `application_name` defaults to `kleos-pg`.
pub fn parse(url: &str) -> Result<Config> {
    let mut c = Config::from_str(url).map_err(|e| miette!("pg connection string: {e}"))?;
    if c.get_application_name().is_none() {
        c.application_name("kleos-pg");
    }
    Ok(c)
}

/// Connect; the connection task runs on the current runtime.
pub async fn connect(cfg: &Config) -> Result<Client> {
    match cfg.get_ssl_mode() {
        SslMode::Disable => {
            let (client, conn) = cfg.connect(tokio_postgres::NoTls).await.map_err(pg_err)?;
            tokio::spawn(async move {
                if let Err(e) = conn.await {
                    log::debug!("pg connection closed: {e}");
                }
            });
            Ok(client)
        }
        mode => connect_tls(cfg, mode).await,
    }
}

#[cfg(feature = "tls")]
async fn connect_tls(cfg: &Config, mode: SslMode) -> Result<Client> {
    let mut b = native_tls::TlsConnector::builder();
    match mode {
        // libpq semantics: require encrypts without checking the certificate
        SslMode::Prefer | SslMode::Require => {
            b.danger_accept_invalid_certs(true).danger_accept_invalid_hostnames(true);
        }
        SslMode::VerifyCa => {
            b.danger_accept_invalid_hostnames(true);
        }
        _ => {}
    }
    if let Some(pem) = cfg.get_ssl_root_cert() {
        let cert = native_tls::Certificate::from_pem(pem).map_err(|e| miette!("sslrootcert: {e}"))?;
        b.add_root_certificate(cert);
    }
    let tls = postgres_native_tls::MakeTlsConnector::new(b.build().map_err(|e| miette!("tls: {e}"))?);
    let (client, conn) = cfg.connect(tls).await.map_err(pg_err)?;
    tokio::spawn(async move {
        if let Err(e) = conn.await {
            log::debug!("pg connection closed: {e}");
        }
    });
    Ok(client)
}

#[cfg(not(feature = "tls"))]
async fn connect_tls(_cfg: &Config, _mode: SslMode) -> Result<Client> {
    miette::bail!("TLS needs kleos-pg built with the `tls` feature (or sslmode=disable)")
}

/// A logical replication connection (`replication=database`).
pub async fn connect_replication(cfg: &Config) -> Result<Client> {
    let mut c = cfg.clone();
    c.replication_mode(ReplicationMode::Logical);
    connect(&c).await
}

/// `SHOW server_version_num` as a number (170011 for 17.11).
pub async fn server_version(client: &Client) -> Result<i32> {
    let rows = client.simple_query("SHOW server_version_num").await.map_err(pg_err)?;
    for m in rows {
        if let tokio_postgres::SimpleQueryMessage::Row(r) = m {
            return r.get(0).and_then(|v| v.parse().ok()).ok_or_else(|| miette!("bad server_version_num"));
        }
    }
    Err(miette!("no server_version_num"))
}

/// Rows of a simple (text) query as strings.
pub async fn simple_rows(client: &Client, sql: &str) -> Result<Vec<Vec<Option<String>>>> {
    let msgs = client.simple_query(sql).await.map_err(pg_err)?;
    let mut out = vec![];
    for m in msgs {
        if let tokio_postgres::SimpleQueryMessage::Row(r) = m {
            out.push((0..r.len()).map(|i| r.get(i).map(|s| s.to_string())).collect());
        }
    }
    Ok(out)
}

/// Quote an identifier.
pub fn ident(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

/// Quote `schema.table`.
pub fn qident(qualified: &str) -> String {
    match qualified.split_once('.') {
        Some((s, t)) => format!("{}.{}", ident(s), ident(t)),
        None => ident(qualified),
    }
}

/// Quote a string literal.
pub fn lit(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// `0/16B3748` ↔ u64.
pub fn parse_lsn(s: &str) -> Option<u64> {
    let (a, b) = s.split_once('/')?;
    Some((u64::from_str_radix(a, 16).ok()? << 32) | u64::from_str_radix(b, 16).ok()?)
}

pub fn fmt_lsn(l: u64) -> String {
    format!("{:X}/{:X}", l >> 32, l & 0xffff_ffff)
}

#[cfg(test)]
mod tests {
    #[test]
    fn lsn_round_trip() {
        assert_eq!(super::parse_lsn("0/16B3748"), Some(0x16B3748));
        assert_eq!(super::fmt_lsn(0x1_0000_00FF), "1/FF");
        assert_eq!(super::qident("public.my\"t"), "\"public\".\"my\"\"t\"");
    }
}
