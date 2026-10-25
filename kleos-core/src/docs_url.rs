/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Documents by URL (feature `docs-url`, off by default; docs/specs/documents.md,
//! "Documents by URL"). Only `http`/`https`; every hop's host must resolve to
//! public addresses only (no loopback, private, link-local, CGNAT, unique-local,
//! multicast or unspecified ones), redirects are followed by hand and checked
//! again, the body is capped and the content type must be a document type.
//!
//! Each hop resolves its host once; the connection is pinned to exactly the
//! addresses that passed the check (the HTTP client's resolver returns them
//! and never asks DNS again), so a DNS server that changes its answer
//! between the check and the connection (rebinding) cannot redirect it.

use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};

use miette::{bail, miette, Result};

/// Default body cap (bytes); `KLEOS_DOC_URL_MAX_BYTES` overrides it.
pub const DEFAULT_MAX_BYTES: usize = 100 * 1024 * 1024;
const TIMEOUT_SECS: u64 = 30;
const MAX_REDIRECTS: usize = 5;

/// Content types accepted (prefix match); anything else is refused before
/// the body is read.
const DOC_TYPES: &[&str] = &[
    "application/pdf",
    "image/",
    "text/",
    "application/xhtml",
    "application/vnd.openxmlformats-officedocument.wordprocessingml",
    "application/octet-stream",
];

/// A fetched document: bytes, the final URL, the content type and a file name.
#[derive(Debug)]
pub struct Fetched {
    pub bytes: Vec<u8>,
    pub url: String,
    pub mime: Option<String>,
    pub name: String,
}

fn max_bytes() -> usize {
    std::env::var("KLEOS_DOC_URL_MAX_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_MAX_BYTES)
}

/// Whether an address is safe to fetch from (public unicast only).
pub fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => public_v4(v4),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return public_v4(v4);
            }
            let seg = v6.segments();
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (seg[0] & 0xfe00) == 0xfc00 // unique local fc00::/7
                || (seg[0] & 0xffc0) == 0xfe80 // link local fe80::/10
                || (seg[0] == 0x2001 && seg[1] == 0x0db8) // documentation
                || v6 == Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0, 0))
        }
    }
}

fn public_v4(v4: Ipv4Addr) -> bool {
    let o = v4.octets();
    !(v4.is_private()
        || v4.is_loopback()
        || v4.is_link_local()
        || v4.is_broadcast()
        || v4.is_unspecified()
        || v4.is_multicast()
        || v4.is_documentation()
        || o[0] == 0
        || (o[0] == 100 && (64..128).contains(&o[1])) // CGNAT 100.64/10
        || (o[0] == 192 && o[1] == 0 && o[2] == 0) // IETF 192.0.0/24
        || (o[0] == 198 && (18..20).contains(&o[1])) // benchmarking
        || o[0] >= 240)
}

/// Scheme, host and port of an `http(s)` URL.
fn parse(url: &str) -> Result<(bool, String, u16)> {
    let (https, rest) = if let Some(r) = url.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = url.strip_prefix("http://") {
        (false, r)
    } else {
        bail!("documents by URL: only http and https URLs are allowed")
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if authority.contains('@') {
        bail!("documents by URL: credentials in the URL are not allowed");
    }
    let (host, port) = if let Some(h) = authority.strip_prefix('[') {
        let (h, tail) = h.split_once(']').ok_or_else(|| miette!("documents by URL: bad IPv6 host"))?;
        let port = tail.strip_prefix(':').map(|p| p.parse::<u16>()).transpose().map_err(|_| miette!("documents by URL: bad port"))?;
        (h.to_string(), port)
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), Some(p.parse::<u16>().map_err(|_| miette!("documents by URL: bad port"))?)),
            None => (authority.to_string(), None),
        }
    };
    if host.is_empty() {
        bail!("documents by URL: no host");
    }
    Ok((https, host, port.unwrap_or(if https { 443 } else { 80 })))
}

/// Resolve the host once and refuse it unless every address passes `allow`;
/// the addresses returned are the only ones the connection may use.
fn check_host(
    host: &str,
    port: u16,
    resolve: &dyn Fn(&str, u16) -> std::io::Result<Vec<IpAddr>>,
    allow: &dyn Fn(IpAddr) -> bool,
) -> Result<Vec<SocketAddr>> {
    let addrs: Vec<IpAddr> = match host.parse::<IpAddr>() {
        Ok(ip) => vec![ip],
        Err(_) => resolve(host, port).map_err(|e| miette!("documents by URL: cannot resolve {host}: {e}"))?,
    };
    if addrs.is_empty() {
        bail!("documents by URL: {host} has no addresses");
    }
    if let Some(bad) = addrs.iter().find(|ip| !allow(**ip)) {
        bail!("documents by URL: {host} resolves to a non-public address ({bad}); refused");
    }
    Ok(addrs.into_iter().map(|ip| SocketAddr::new(ip, port)).collect())
}

fn system_resolve(host: &str, port: u16) -> std::io::Result<Vec<IpAddr>> {
    Ok((host, port).to_socket_addrs()?.map(|a| a.ip()).collect())
}

fn join(base: &str, loc: &str) -> String {
    if loc.starts_with("http://") || loc.starts_with("https://") {
        return loc.to_string();
    }
    let (scheme, rest) = base.split_once("://").unwrap_or(("https", base));
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if let Some(p) = loc.strip_prefix("//") {
        return format!("{scheme}://{p}");
    }
    if loc.starts_with('/') {
        return format!("{scheme}://{authority}{loc}");
    }
    let path = &rest[authority.len()..];
    let dir = path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
    format!("{scheme}://{authority}{dir}/{loc}")
}

/// Fetch a document. Redirects are followed by hand so each hop's host is
/// checked (and its connection pinned); the body stops at the cap.
pub fn fetch(url: &str) -> Result<Fetched> {
    fetch_with(url, &system_resolve, &is_public)
}

/// [`fetch`] with the resolver and the address check given (tests).
pub(crate) fn fetch_with(
    url: &str,
    resolve: &dyn Fn(&str, u16) -> std::io::Result<Vec<IpAddr>>,
    allow: &dyn Fn(IpAddr) -> bool,
) -> Result<Fetched> {
    let cap = max_bytes();
    let mut cur = url.to_string();
    for _ in 0..=MAX_REDIRECTS {
        let (_https, host, port) = parse(&cur)?;
        let pinned = check_host(&host, port, resolve, allow)?;
        // the agent of this hop resolves every name to the checked addresses
        let agent = ureq::AgentBuilder::new()
            .redirects(0)
            .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
            .resolver(move |_netloc: &str| -> std::io::Result<Vec<SocketAddr>> { Ok(pinned.clone()) })
            .build();
        let resp = match agent.get(&cur).set("User-Agent", "kleos-docs").call() {
            Ok(r) => r,
            Err(ureq::Error::Status(code, _)) => bail!("documents by URL: {cur} answered HTTP {code}"),
            Err(e) => bail!("documents by URL: {cur}: {e}"),
        };
        let code = resp.status();
        if (300..400).contains(&code) {
            let loc = resp.header("location").ok_or_else(|| miette!("documents by URL: redirect without a Location"))?;
            cur = join(&cur, loc);
            continue;
        }
        if !(200..300).contains(&code) {
            bail!("documents by URL: {cur} answered HTTP {code}");
        }
        let mime = resp.header("content-type").map(|c| c.split(';').next().unwrap_or("").trim().to_lowercase());
        if let Some(m) = &mime {
            if !m.is_empty() && !DOC_TYPES.iter().any(|t| m.starts_with(t)) {
                bail!("documents by URL: {cur} is '{m}', not a document type");
            }
        }
        if let Some(len) = resp.header("content-length").and_then(|l| l.trim().parse::<usize>().ok()) {
            if len > cap {
                bail!("documents by URL: {cur} is {len} bytes, over the {cap}-byte limit");
            }
        }
        let mut bytes = Vec::new();
        resp.into_reader()
            .take(cap as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| miette!("documents by URL: {cur}: {e}"))?;
        if bytes.len() > cap {
            bail!("documents by URL: {cur} is over the {cap}-byte limit");
        }
        let name = cur
            .split(['?', '#'])
            .next()
            .and_then(|p| p.rsplit('/').next())
            .filter(|n| !n.is_empty())
            .unwrap_or("document")
            .to_string();
        return Ok(Fetched { bytes, url: cur, mime: mime.filter(|m| m != "application/octet-stream" && !m.is_empty()), name });
    }
    bail!("documents by URL: more than {MAX_REDIRECTS} redirects")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_non_public_and_bad_urls() {
        for ip in ["127.0.0.1", "10.1.2.3", "192.168.0.1", "172.16.5.4", "169.254.169.254", "100.64.0.1", "0.0.0.0", "::1", "fe80::1", "fd00::1", "::ffff:10.0.0.1"] {
            assert!(!is_public(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["93.184.216.34", "1.1.1.1", "2606:4700:4700::1111"] {
            assert!(is_public(ip.parse().unwrap()), "{ip}");
        }
        assert!(fetch("file:///etc/passwd").is_err());
        assert!(fetch("ftp://example.com/x.pdf").is_err());
        assert!(fetch("http://user:pw@example.com/x").is_err());
        let e = fetch("http://127.0.0.1:9/x.pdf").unwrap_err().to_string();
        assert!(e.contains("non-public"), "{e}");
        let e = fetch("http://localhost/x.pdf").unwrap_err().to_string();
        assert!(e.contains("non-public"), "{e}");
        assert_eq!(join("https://a.com/x/y.pdf", "/z.pdf"), "https://a.com/z.pdf");
        assert_eq!(join("https://a.com/x/y.pdf", "z.pdf"), "https://a.com/x/z.pdf");
    }

    /// An HTTP server on loopback answering one connection per reply, in
    /// order; `replies` gets the server's port.
    fn serve(replies: impl FnOnce(u16) -> Vec<String>) -> u16 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let replies = replies(port);
        std::thread::spawn(move || {
            for r in replies {
                let (mut c, _) = l.accept().unwrap();
                let mut buf = [0u8; 4096];
                let _ = std::io::Read::read(&mut c, &mut buf);
                let _ = std::io::Write::write_all(&mut c, r.as_bytes());
            }
        });
        port
    }

    #[test]
    fn connections_are_pinned_to_the_checked_address() {
        let body = "%PDF-1.4 tiny";
        // a.test redirects to b.test; each name resolves to loopback on its
        // first lookup and to another address on any later one (a rebinding
        // DNS server): the fetch works only if each hop looks up once and
        // connects to the address it checked
        let port = serve(|port| {
            vec![
                format!("HTTP/1.1 302 Found\r\nLocation: http://b.test:{port}/doc.pdf\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"),
                format!("HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()),
            ]
        });
        let calls: std::sync::Mutex<Vec<String>> = Default::default();
        let resolve = |host: &str, _port: u16| -> std::io::Result<Vec<IpAddr>> {
            let mut c = calls.lock().unwrap();
            let first = !c.iter().any(|h| h == host);
            c.push(host.to_string());
            Ok(vec![if first { "127.0.0.1" } else { "10.0.0.66" }.parse().unwrap()])
        };
        // loopback stands in for a public address here
        let allow = |ip: IpAddr| ip.is_loopback();
        let f = fetch_with(&format!("http://a.test:{port}/start"), &resolve, &allow).unwrap();
        assert_eq!(f.bytes, body.as_bytes());
        assert_eq!(f.name, "doc.pdf");
        assert_eq!(*calls.lock().unwrap(), vec!["a.test".to_string(), "b.test".to_string()], "one lookup per hop");
        // a name that resolves to a non-public address is refused before any
        // connection
        let e = fetch_with("http://evil.test:9/x.pdf", &|_, _| Ok(vec!["10.0.0.5".parse().unwrap()]), &is_public).unwrap_err().to_string();
        assert!(e.contains("non-public"), "{e}");
    }
}
