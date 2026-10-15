/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Read-only access to the local Hugging Face cache (`hub/models--org--name`).
//! Models are optional runtime downloads made by the user (e.g. `hf download`);
//! this crate never fetches anything.

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
        "{repo}/{file} (looked in {}; fetch it with `hf download {repo}`)",
        root.display()
    )))
}
