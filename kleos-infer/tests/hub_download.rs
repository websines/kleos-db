/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Model download into an empty Hugging Face cache (network; `--ignored`).
//! One test, because it changes process-wide environment variables.

use kleos_infer::*;

#[test]
#[ignore]
fn enable_downloads_missing_model_unless_offline() {
    let repo = "minishlab/potion-base-8M";
    let cache = std::env::temp_dir().join(format!("kleos-hub-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    std::fs::create_dir_all(&cache).unwrap();
    std::env::set_var("HF_HUB_CACHE", &cache);

    // offline: nothing is fetched, and the error says how to get the model
    std::env::set_var("HF_HUB_OFFLINE", "1");
    let e = StaticEmbedder::new(repo).err().expect("offline enable must fail on an empty cache");
    let msg = e.to_string();
    assert!(msg.contains("hf download") && msg.contains("HF_HUB_OFFLINE"), "{msg}");
    assert!(resolve_hf_file(repo, "config.json").is_err());

    // online: enabling fetches every file into the standard cache layout
    std::env::remove_var("HF_HUB_OFFLINE");
    let m = StaticEmbedder::new(repo).unwrap();
    for f in ["config.json", "tokenizer.json", "model.safetensors"] {
        let p = resolve_hf_file(repo, f).unwrap();
        assert!(p.starts_with(&cache), "{p:?}");
    }
    let root = cache.join("models--minishlab--potion-base-8M");
    assert!(root.join("refs/main").is_file());
    let v = m.embed(&["hello world"]).unwrap();
    assert_eq!(v.len(), 1);
    assert!(v[0].iter().any(|x| *x != 0.0));

    // a second enable is served from the cache: no network needed
    std::env::set_var("HF_HUB_OFFLINE", "1");
    StaticEmbedder::new(repo).unwrap();
    std::env::remove_var("HF_HUB_OFFLINE");

    // a repo that doesn't exist fails with a download error naming it
    let e = StaticEmbedder::new("kleos-test/no-such-model-xyz").err().unwrap().to_string();
    assert!(e.contains("kleos-test/no-such-model-xyz"), "{e}");

    let _ = std::fs::remove_dir_all(&cache);
}
