/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The CLI / MCP config file (kleos fork; docs/specs/mcp-cli-skills.md):
//! `$KLEOS_CONFIG`, else `~/.config/kleos/config.json`. Every field is
//! optional and command-line flags override it.
//!
//! ```json
//! {
//!   "db": "~/.local/share/kleos/memory.db", "engine": "sqlite",
//!   "url": null, "token": null, "ns": "default",
//!   "embedders": ["gemma=embeddinggemma"],
//!   "setup": ["::decider enable nano"],
//!   "init": {"embedder": "gemma", "decider": "nano"}
//! }
//! ```

use std::path::PathBuf;

use serde_derive::{Deserialize, Serialize};
use serde_json::Value;

use crate::backend::{Area, Backend};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Embedded store path.
    pub db: Option<String>,
    /// Storage engine of the embedded store (`sqlite` by default; `mem`,
    /// `rocksdb` when built in).
    pub engine: Option<String>,
    /// A running kleos server to talk to instead of an embedded store.
    pub url: Option<String>,
    /// That server's auth token (`x-cozo-auth`).
    pub token: Option<String>,
    /// Default namespace.
    pub ns: Option<String>,
    /// Built-in embedders to enable in embedded mode, as for
    /// `kleos server --embedder`: `name=model[:precision[:device[:dim]]]`.
    pub embedders: Vec<String>,
    /// CozoScript system ops run once at start in embedded mode, e.g.
    /// `::decider enable nano` or `::extractor enable gliner`.
    pub setup: Vec<String>,
    /// `::memory init` options applied at start in embedded mode
    /// (`embedder`, `decider`, `extractor`, `ontology`, ...).
    pub init: Option<Value>,
}

/// Where the config file lives.
pub fn config_path(explicit: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(expand(p));
    }
    if let Ok(p) = std::env::var("KLEOS_CONFIG") {
        if !p.trim().is_empty() {
            return Some(expand(&p));
        }
    }
    home().map(|h| h.join(".config").join("kleos").join("config.json"))
}

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// `~/` expansion.
pub fn expand(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), home()) {
        (Some(rest), Some(h)) => h.join(rest),
        _ => PathBuf::from(p),
    }
}

impl Config {
    /// Reads the config file. A missing default file is an empty config; a
    /// missing explicit file or a malformed one is an error.
    pub fn load(explicit: Option<&str>) -> Result<Self, String> {
        let Some(path) = config_path(explicit) else { return Ok(Self::default()) };
        match std::fs::read_to_string(&path) {
            Ok(s) => serde_json::from_str(&s).map_err(|e| format!("config {}: {e}", path.display())),
            Err(_) if explicit.is_none() => Ok(Self::default()),
            Err(e) => Err(format!("config {}: {e}", path.display())),
        }
    }
}

/// Connection options after merging flags over the config file.
#[derive(Clone, Debug, Default)]
pub struct Target {
    pub db: Option<String>,
    pub engine: Option<String>,
    pub url: Option<String>,
    pub token: Option<String>,
    pub ns: Option<String>,
    pub embedders: Vec<String>,
}

impl Target {
    pub fn ns(&self, cfg: &Config) -> String {
        self.ns.clone().or_else(|| cfg.ns.clone()).unwrap_or_else(|| "default".to_string())
    }

    /// Opens the backend: a remote server when a URL is given, else the
    /// embedded store; in embedded mode it also enables the configured models
    /// and applies `init`.
    pub fn connect(&self, cfg: &Config) -> Result<Backend, String> {
        if let Some(url) = self.url.clone().or_else(|| cfg.url.clone()) {
            return Ok(Backend::remote(&url, self.token.clone().or_else(|| cfg.token.clone())));
        }
        let Some(db) = self.db.clone().or_else(|| cfg.db.clone()) else {
            return Err("no store configured: pass --db PATH (or --url URL for a running kleos server), \
                        or set \"db\" in ~/.config/kleos/config.json"
                .to_string());
        };
        let engine = self.engine.clone().or_else(|| cfg.engine.clone()).unwrap_or_else(|| "sqlite".to_string());
        let path = expand(&db);
        if engine != "mem" {
            if let Some(dir) = path.parent() {
                if !dir.as_os_str().is_empty() {
                    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
                }
            }
        }
        let backend = Backend::open(&engine, &path.to_string_lossy())?;
        let Backend::Embedded(inst) = &backend else { unreachable!() };
        for arg in self.embedders.iter().chain(cfg.embedders.iter()) {
            let (name, spec) = kleos::EmbedderSpec::parse_cli(arg).map_err(|e| format!("embedder '{arg}': {e}"))?;
            kleos::enable_embedder(&name, spec).map_err(|e| format!("embedder '{arg}': {e}"))?;
        }
        for op in &cfg.setup {
            inst.run_script(op, Default::default(), kleos::ScriptMutability::Mutable)
                .map_err(|e| format!("config setup `{op}`: {e}"))?;
        }
        if let Some(init) = &cfg.init {
            backend.call(Area::Memory, "init", init.clone()).map_err(|e| format!("config init: {e}"))?;
        }
        Ok(backend)
    }
}
