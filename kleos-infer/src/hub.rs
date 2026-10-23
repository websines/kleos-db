/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The local Hugging Face cache (`hub/models--org--name`), shared with the
//! Python `huggingface_hub` tools.
//!
//! Model constructors call [`ensure_hf_files`] when a model is enabled (never
//! during a query): with the `download` feature (on by default) missing files
//! are fetched into the cache, unless `HF_HUB_OFFLINE` is set. Without it, or
//! offline, files must be put there beforehand, e.g. with `hf download`.

use std::path::PathBuf;

use crate::{Error, Result};

/// The `hub` directory of the HF cache: `$HF_HUB_CACHE`, else
/// `$HF_HOME/hub`, else `~/.cache/huggingface/hub`.
pub fn hf_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("HF_HOME") {
        return PathBuf::from(p).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/huggingface/hub")
}

/// Whether downloads are switched off (`HF_HUB_OFFLINE=1`, as in the Python
/// tools).
pub fn hf_offline() -> bool {
    std::env::var("HF_HUB_OFFLINE")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Make sure model `repo` has `files` in the local cache, downloading them
/// (and `optional` files, which the repo may lack) when any is missing.
/// Local directories, and every repo when offline or built without the
/// `download` feature, are left as they are; a later
/// [`resolve_hf_file`] then reports what is missing.
pub fn ensure_hf_files(repo: &str, files: &[&str], optional: &[&str]) -> Result<()> {
    if PathBuf::from(repo).is_dir() || files.iter().all(|f| resolve_hf_file(repo, f).is_ok()) {
        return Ok(());
    }
    if hf_offline() {
        return Ok(());
    }
    download(repo, files, optional)
}

#[cfg(feature = "download")]
fn download(repo: &str, files: &[&str], optional: &[&str]) -> Result<()> {
    use hf_hub::api::sync::ApiBuilder;
    // `HF_TOKEN` first, else the token file `hf auth login` writes; needed
    // for gated repos such as google/embeddinggemma-300m.
    let token = std::env::var("HF_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty())
        .or_else(|| hf_hub::Cache::from_env().token());
    let mut builder = ApiBuilder::from_cache(hf_hub::Cache::new(hf_cache_dir()))
        .with_progress(false)
        .with_token(token);
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        builder = builder.with_endpoint(endpoint);
    }
    let api = builder.build().map_err(|e| Error::Download(format!("{repo}: {e}")))?;
    let model = api.model(repo.to_string());
    for f in files {
        if resolve_hf_file(repo, f).is_err() {
            model.get(f).map_err(|e| {
                Error::Download(format!(
                    "{repo}/{f}: {e} (gated repos need their licence accepted on huggingface.co and HF_TOKEN set; \
                     or fetch it with `hf download {repo}`)"
                ))
            })?;
        }
    }
    for f in optional {
        if resolve_hf_file(repo, f).is_err() {
            // absent from this repo: fine
            let _ = model.get(f);
        }
    }
    Ok(())
}

#[cfg(not(feature = "download"))]
fn download(_repo: &str, _files: &[&str], _optional: &[&str]) -> Result<()> {
    Ok(())
}

/// Path of `file` in the cached snapshot of model `repo` (`org/name`), or of
/// `repo/file` when `repo` is a local directory. Uses `refs/main` when present,
/// else any snapshot holding the file.
pub fn resolve_hf_file(repo: &str, file: &str) -> Result<PathBuf> {
    let local = PathBuf::from(repo);
    if local.is_dir() {
        let p = local.join(file);
        return if p.exists() {
            Ok(p)
        } else {
            Err(Error::NotFound(p.display().to_string()))
        };
    }
    let root = hf_cache_dir().join(format!("models--{}", repo.replace('/', "--")));
    if let Ok(rev) = std::fs::read_to_string(root.join("refs/main")) {
        let p = root.join("snapshots").join(rev.trim()).join(file);
        if p.exists() {
            return Ok(p);
        }
    }
    if let Ok(dir) = std::fs::read_dir(root.join("snapshots")) {
        for e in dir.flatten() {
            let p = e.path().join(file);
            if p.exists() {
                return Ok(p);
            }
        }
    }
    Err(Error::NotFound(format!(
        "{repo}/{file} (looked in {}; fetch it with `hf download {repo}`{})",
        root.display(),
        if hf_offline() { ", or unset HF_HUB_OFFLINE" } else { "" }
    )))
}
