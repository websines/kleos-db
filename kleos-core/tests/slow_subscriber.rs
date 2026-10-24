/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Concurrency stress: a change-feed subscriber that never drains its
//! (bounded) channel must not block writers. Many concurrent `remember`s
//! run while such a subscriber is attached to the memory relation; they must
//! all finish. On main before the callback fix, writers block after the
//! first commit (`send_callbacks_inner` waits on the full bounded channel).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use kleos::embed::{self, Embedder};
use kleos::memory::{InitOpts, RememberOpts};
use kleos::DbInstance;

struct Words;

impl Embedder for Words {
    fn embed(&self, texts: &[&str]) -> miette::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; 32];
                for (i, b) in t.bytes().enumerate() {
                    v[(b as usize + i) % 32] += 1.0;
                }
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                v.iter().map(|x| x / n).collect()
            })
            .collect())
    }
    fn dim(&self) -> usize {
        32
    }
}

#[test]
fn a_slow_subscriber_does_not_block_writers() {
    embed::register_embedder("slow_sub_words", Arc::new(Words)).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    db.memory_init(InitOpts { embedder: Some("slow_sub_words".into()), ..Default::default() }).unwrap();
    // capacity 1, never read
    let (_id, _rx) = db.register_callback("kleos_mem", Some(1));
    let done = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel();
    for t in 0..8 {
        let (db, done, tx) = (db.clone(), done.clone(), tx.clone());
        std::thread::spawn(move || {
            for i in 0..25 {
                let text = format!("writer {t} fact {i}: the harbour crane {i} moved to berth {t}");
                db.remember(&text, RememberOpts { ns: Some(format!("w{t}")), ..Default::default() }).unwrap();
                done.fetch_add(1, Ordering::Relaxed);
            }
            let _ = tx.send(());
        });
    }
    drop(tx);
    for _ in 0..8 {
        rx.recv_timeout(Duration::from_secs(60))
            .unwrap_or_else(|_| panic!("writers blocked: {} of 200 remembers done", done.load(Ordering::Relaxed)));
    }
    assert_eq!(done.load(Ordering::Relaxed), 200);
}
