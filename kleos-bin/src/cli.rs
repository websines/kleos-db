/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The `kleos` command-line tool (kleos fork; docs/specs/mcp-cli-skills.md):
//! memory, documents, ontology, workflows and branches from a shell, against
//! an embedded store (`--db`) or a running server (`--url`). `--json` prints
//! machine-readable output for agents.

use std::io::{IsTerminal, Write};

use clap::{Args, Subcommand};
use serde_json::{json, Value};

use crate::backend::{Area, Backend};
use crate::config::{Config, Target};

/// Connection and output options shared by every CLI command.
#[derive(Args, Clone, Debug, Default)]
pub struct ConnArgs {
    /// Embedded store path (SQLite unless --engine)
    #[arg(long)]
    pub db: Option<String>,
    /// Storage engine of the embedded store: sqlite (default), mem, rocksdb
    #[arg(long)]
    pub engine: Option<String>,
    /// A running kleos server instead of an embedded store
    #[arg(long)]
    pub url: Option<String>,
    /// The server's auth token
    #[arg(long)]
    pub token: Option<String>,
    /// Namespace (default: config "ns", else "default")
    #[arg(long)]
    pub ns: Option<String>,
    /// Machine-readable JSON output
    #[arg(long)]
    pub json: bool,
    /// Config file (default $KLEOS_CONFIG or ~/.config/kleos/config.json)
    #[arg(long = "config")]
    pub config: Option<String>,
    /// Built-in embedder to enable in embedded mode, `name=model[:precision[:device[:dim]]]` (repeatable)
    #[arg(long = "embedder")]
    pub embedders: Vec<String>,
}

/// The CLI commands.
#[derive(Subcommand, Debug)]
pub enum CliCmd {
    /// Store a statement (and attached files)
    Remember {
        text: String,
        /// Attach a file (repeatable)
        #[arg(long = "attach")]
        attach: Vec<String>,
        /// Entity the statement is about (repeatable; found automatically when omitted)
        #[arg(long = "entity")]
        entities: Vec<String>,
        /// When it was said, microseconds since the epoch
        #[arg(long)]
        at: Option<i64>,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Search memory; prints the context block unless --list
    Recall {
        query: String,
        /// Number of hits
        #[arg(short, long)]
        k: Option<usize>,
        /// Print hits instead of the context block
        #[arg(long)]
        list: bool,
        /// Token budget of the context block
        #[arg(long)]
        budget: Option<usize>,
        /// Recall as of this time (microseconds since the epoch)
        #[arg(long)]
        as_of: Option<i64>,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Forget memories by id (retract; --erase deletes them everywhere)
    Forget {
        #[arg(required = true)]
        ids: Vec<String>,
        #[arg(long)]
        erase: bool,
        /// Do not ask for confirmation
        #[arg(long)]
        yes: bool,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Pending replacements, possible contradictions and recent changes
    Reflect {
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// What memory knows about an entity
    Profile {
        name: String,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Documents
    Doc {
        #[command(subcommand)]
        cmd: DocCmd,
    },
    /// The namespace's ontology
    Ontology {
        #[command(subcommand)]
        cmd: OntologyCmd,
    },
    /// Durable workflows
    Flow {
        #[command(subcommand)]
        cmd: FlowCmd,
    },
    /// Memory branches
    Branch {
        #[command(subcommand)]
        cmd: BranchCmd,
    },
    /// Run a read-only CozoScript query
    Query {
        script: String,
        /// Parameters as a JSON object
        #[arg(long)]
        params: Option<String>,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Serve MCP (protocol 2026-07-28) over stdio
    Mcp {
        /// Offer the read-only `query` tool
        #[arg(long)]
        allow_query: bool,
        #[command(flatten)]
        conn: ConnArgs,
    },
}

#[derive(Subcommand, Debug)]
pub enum DocCmd {
    /// Store a document
    Add {
        file: String,
        /// A note stored with it
        #[arg(long)]
        text: Option<String>,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// One document's status
    Status {
        doc: String,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// The namespace's documents
    List {
        #[command(flatten)]
        conn: ConnArgs,
    },
}

#[derive(Subcommand, Debug)]
pub enum OntologyCmd {
    /// Ontologies and which namespaces use them
    List {
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Learn properties from untyped memories now
    Learn {
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Changes the ontology made to itself
    History {
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Undo one change from the history
    Undo {
        change: String,
        #[command(flatten)]
        conn: ConnArgs,
    },
}

#[derive(Subcommand, Debug)]
pub enum FlowCmd {
    /// Start a workflow
    Start {
        workflow: String,
        /// Input as JSON
        #[arg(long)]
        input: Option<String>,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// A workflow's status
    Status {
        id: String,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Send a workflow a signal
    Signal {
        id: String,
        name: String,
        /// Payload as JSON
        #[arg(long)]
        payload: Option<String>,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Cancel a workflow
    Cancel {
        id: String,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// List workflows
    List {
        #[arg(long)]
        status: Option<String>,
        #[command(flatten)]
        conn: ConnArgs,
    },
}

#[derive(Subcommand, Debug)]
pub enum BranchCmd {
    /// Fork a namespace (instant copy)
    Fork {
        to: String,
        /// Namespace to fork (default --ns)
        #[arg(long)]
        from: Option<String>,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// What differs between two namespaces
    Diff {
        a: String,
        b: String,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Merge a branch into another namespace (default: its parent)
    Merge {
        from: String,
        #[arg(long)]
        into: Option<String>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        yes: bool,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Rewind a namespace to a time (microseconds since the epoch; undoable)
    Rewind {
        target: String,
        at: i64,
        #[arg(long)]
        yes: bool,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Discard a branch
    Discard {
        target: String,
        #[arg(long)]
        yes: bool,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// List branches
    List {
        #[command(flatten)]
        conn: ConnArgs,
    },
}

struct Ctx {
    backend: Backend,
    ns: String,
    json: bool,
}

fn connect(c: &ConnArgs) -> Result<(Ctx, Config), String> {
    let cfg = Config::load(c.config.as_deref())?;
    let target = Target {
        db: c.db.clone(),
        engine: c.engine.clone(),
        url: c.url.clone(),
        token: c.token.clone(),
        ns: c.ns.clone(),
        embedders: c.embedders.clone(),
    };
    let ns = target.ns(&cfg);
    let backend = target.connect(&cfg)?;
    Ok((Ctx { backend, ns, json: c.json }, cfg))
}

fn parse_json(label: &str, s: Option<&str>) -> Result<Value, String> {
    match s {
        None => Ok(Value::Null),
        Some(s) => serde_json::from_str(s).map_err(|e| format!("{label} is not valid JSON: {e}")),
    }
}

fn confirm(question: &str, yes: bool) -> Result<(), String> {
    if yes {
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        return Err(format!("{question} Refusing without --yes (not a terminal)."));
    }
    eprint!("{question} [y/N] ");
    std::io::stderr().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).map_err(|e| e.to_string())?;
    if matches!(line.trim().to_lowercase().as_str(), "y" | "yes") {
        Ok(())
    } else {
        Err("cancelled; nothing was changed".into())
    }
}

/// Prints a result: JSON with --json, else a short human form.
fn print(ctx: &Ctx, v: &Value, human: impl FnOnce(&Value) -> Option<String>) {
    if ctx.json {
        println!("{}", json!({"ok": true, "result": v}));
    } else {
        match human(v) {
            Some(s) => println!("{s}"),
            None => println!("{}", serde_json::to_string_pretty(v).unwrap_or_default()),
        }
    }
}

/// The ids a `remember` result added or replaced, for the human form.
fn remember_summary(v: &Value) -> Option<String> {
    let mut out = vec![];
    if let Some(id) = v.get("id").and_then(|x| x.as_str()) {
        out.push(format!("remembered {id}"));
    }
    if let Some(a) = v.get("action").and_then(|x| x.as_str()) {
        if out.is_empty() {
            out.push(a.to_string());
        }
    }
    if let Some(s) = v.get("superseded").and_then(|x| x.as_array()).filter(|a| !a.is_empty()) {
        out.push(format!("replaced {}", s.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")));
    }
    if let Some(docs) = v.get("documents").and_then(|x| x.as_array()) {
        for d in docs {
            out.push(format!(
                "document {} ({}): {}, {} pages, {} chunks",
                d.get("doc").and_then(|x| x.as_str()).unwrap_or("?"),
                d.get("name").and_then(|x| x.as_str()).unwrap_or("?"),
                d.get("status").and_then(|x| x.as_str()).unwrap_or("?"),
                d.get("pages").and_then(|x| x.as_u64()).unwrap_or(0),
                d.get("chunks").and_then(|x| x.as_u64()).unwrap_or(0),
            ));
        }
    }
    (!out.is_empty()).then(|| out.join("\n"))
}

/// Runs a CLI command; the error is printed by the caller.
pub fn run(cmd: CliCmd) -> Result<(), String> {
    match cmd {
        CliCmd::Remember { text, attach, entities, at, conn } => {
            let (ctx, _) = connect(&conn)?;
            let mut a = json!({"ns": ctx.ns, "text": text});
            if !attach.is_empty() {
                a["attachments"] = json!(attach.iter().map(|p| abs(p)).collect::<Vec<_>>());
            }
            if !entities.is_empty() {
                a["entities"] = json!(entities);
            }
            if let Some(at) = at {
                a["at"] = json!(at);
            }
            let v = ctx.backend.call(Area::Memory, "remember", a)?;
            print(&ctx, &v, remember_summary);
        }
        CliCmd::Recall { query, k, list, budget, as_of, conn } => {
            let (ctx, _) = connect(&conn)?;
            let mut a = json!({"ns": ctx.ns, "query": query});
            if !list {
                a["format"] = json!("context");
            }
            for (key, val) in [("k", k.map(|x| x as i64)), ("budget", budget.map(|x| x as i64)), ("as_of", as_of)] {
                if let Some(v) = val {
                    a[key] = json!(v);
                }
            }
            let v = ctx.backend.call(Area::Memory, "recall", a)?;
            print(&ctx, &v, |v| {
                if list {
                    let hits = v.as_array()?;
                    Some(if hits.is_empty() {
                        "(no memories found)".into()
                    } else {
                        hits.iter()
                            .map(|h| format!("{}  {}", h.get("id").and_then(|x| x.as_str()).unwrap_or(""), h.get("text").and_then(|x| x.as_str()).unwrap_or("")))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                } else {
                    let t = v.get("text")?.as_str()?;
                    Some(if t.trim().is_empty() { "(no memories found)".into() } else { t.to_string() })
                }
            });
        }
        CliCmd::Forget { ids, erase, yes, conn } => {
            let (ctx, _) = connect(&conn)?;
            let mode = if erase { "erase" } else { "retract" };
            confirm(&format!("{} {} memor{} in '{}'?", if erase { "Erase (permanently delete)" } else { "Retract" }, ids.len(), if ids.len() == 1 { "y" } else { "ies" }, ctx.ns), yes)?;
            let v = ctx.backend.call(Area::Memory, "forget", json!({"ns": ctx.ns, "ids": ids, "mode": mode}))?;
            print(&ctx, &v, |_| None);
        }
        CliCmd::Reflect { conn } => {
            let (ctx, _) = connect(&conn)?;
            let v = ctx.backend.call(Area::Memory, "reflect", json!({"ns": ctx.ns}))?;
            print(&ctx, &v, |_| None);
        }
        CliCmd::Profile { name, conn } => {
            let (ctx, _) = connect(&conn)?;
            let facts = ctx.backend.call(Area::Ontology, "facts", json!({"ns": ctx.ns, "s": name})).unwrap_or(json!([]));
            let block = ctx.backend.call(Area::Memory, "recall", json!({"ns": ctx.ns, "query": name, "entity": name, "format": "context", "budget": 600}))?;
            let v = json!({"name": name, "facts": facts, "context": block});
            print(&ctx, &v, |v| {
                let mut s = String::new();
                if let Some(fs) = v["facts"].as_array().filter(|a| !a.is_empty()) {
                    s.push_str(&format!("Facts about {name}:\n"));
                    for f in fs {
                        s.push_str(&format!("- {f}\n"));
                    }
                    s.push('\n');
                }
                let t = v["context"]["text"].as_str().unwrap_or_default();
                s.push_str(if t.trim().is_empty() && s.is_empty() { "(nothing known)" } else { t });
                Some(s)
            });
        }
        CliCmd::Doc { cmd } => match cmd {
            DocCmd::Add { file, text, conn } => {
                let (ctx, _) = connect(&conn)?;
                let v = ctx.backend.call(Area::Memory, "remember", json!({"ns": ctx.ns, "text": text.unwrap_or_default(), "attachments": [abs(&file)]}))?;
                print(&ctx, &v, remember_summary);
            }
            DocCmd::Status { doc, conn } => {
                let (ctx, _) = connect(&conn)?;
                let v = ctx.backend.call(Area::Doc, "get", json!({"ns": ctx.ns, "doc": doc}))?;
                if v.is_null() {
                    return Err(format!("no document '{doc}' in namespace '{}'", ctx.ns));
                }
                print(&ctx, &v, |_| None);
            }
            DocCmd::List { conn } => {
                let (ctx, _) = connect(&conn)?;
                let v = ctx.backend.call(Area::Doc, "list", json!({"ns": ctx.ns}))?;
                print(&ctx, &v, |v| {
                    let a = v.as_array()?;
                    Some(if a.is_empty() {
                        "(no documents)".into()
                    } else {
                        a.iter()
                            .map(|d| format!("{}  {}  {}  {} pages", d["doc"].as_str().unwrap_or(""), d["name"].as_str().unwrap_or(""), d["status"].as_str().unwrap_or(""), d["pages"]))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                });
            }
        },
        CliCmd::Ontology { cmd } => {
            let (op, conn, extra) = match cmd {
                OntologyCmd::List { conn } => ("list", conn, None),
                OntologyCmd::Learn { conn } => ("learn", conn, None),
                OntologyCmd::History { conn } => ("history", conn, None),
                OntologyCmd::Undo { change, conn } => ("undo", conn, Some(change)),
            };
            let (ctx, _) = connect(&conn)?;
            let mut a = json!({"ns": ctx.ns});
            if let Some(c) = extra {
                a["change"] = json!(c);
            }
            if op == "list" {
                a = json!({});
            }
            let v = ctx.backend.call(Area::Ontology, op, a)?;
            print(&ctx, &v, |_| None);
        }
        CliCmd::Flow { cmd } => {
            let (op, conn, a) = match cmd {
                FlowCmd::Start { workflow, input, conn } => {
                    let input = parse_json("--input", input.as_deref())?;
                    ("start", conn, json!({"workflow": workflow, "input": input}))
                }
                FlowCmd::Status { id, conn } => ("status", conn, json!({"id": id})),
                FlowCmd::Signal { id, name, payload, conn } => {
                    let payload = parse_json("--payload", payload.as_deref())?;
                    ("signal", conn, json!({"id": id, "name": name, "payload": payload}))
                }
                FlowCmd::Cancel { id, conn } => ("cancel", conn, json!({"id": id})),
                FlowCmd::List { status, conn } => ("list", conn, status.map(|s| json!({"status": s})).unwrap_or(json!({}))),
            };
            let (ctx, _) = connect(&conn)?;
            let v = ctx.backend.call(Area::Flow, op, a)?;
            print(&ctx, &v, |_| None);
        }
        CliCmd::Branch { cmd } => match cmd {
            BranchCmd::Fork { to, from, conn } => {
                let (ctx, _) = connect(&conn)?;
                let v = ctx.backend.call(Area::Memory, "fork", json!({"from": from.unwrap_or_else(|| ctx.ns.clone()), "to": to}))?;
                print(&ctx, &v, |_| None);
            }
            BranchCmd::Diff { a, b, conn } => {
                let (ctx, _) = connect(&conn)?;
                let v = ctx.backend.call(Area::Memory, "diff", json!({"a": a, "b": b}))?;
                print(&ctx, &v, |_| None);
            }
            BranchCmd::Merge { from, into, dry_run, yes, conn } => {
                let (ctx, _) = connect(&conn)?;
                if !dry_run {
                    confirm(&format!("Merge '{from}' into '{}'?", into.clone().unwrap_or_else(|| "its parent".into())), yes)?;
                }
                let mut a = json!({"from": from, "dry_run": dry_run});
                if let Some(i) = into {
                    a["into"] = json!(i);
                }
                let v = ctx.backend.call(Area::Memory, "merge", a)?;
                print(&ctx, &v, |_| None);
            }
            BranchCmd::Rewind { target, at, yes, conn } => {
                let (ctx, _) = connect(&conn)?;
                confirm(&format!("Rewind '{target}' to {at}? (undoable)"), yes)?;
                let v = ctx.backend.call(Area::Memory, "rewind", json!({"ns": target, "to": at}))?;
                print(&ctx, &v, |_| None);
            }
            BranchCmd::Discard { target, yes, conn } => {
                let (ctx, _) = connect(&conn)?;
                confirm(&format!("Discard branch '{target}'?"), yes)?;
                let v = ctx.backend.call(Area::Memory, "discard", json!({"ns": target}))?;
                print(&ctx, &v, |_| None);
            }
            BranchCmd::List { conn } => {
                let (ctx, _) = connect(&conn)?;
                let v = ctx.backend.call(Area::Memory, "branches", json!({}))?;
                print(&ctx, &v, |_| None);
            }
        },
        CliCmd::Query { script, params, conn } => {
            let (ctx, _) = connect(&conn)?;
            let params = parse_json("--params", params.as_deref())?;
            let v = ctx.backend.query(&script, &params)?;
            print(&ctx, &v, |_| None);
        }
        CliCmd::Mcp { allow_query, conn } => {
            let (ctx, _) = connect(&conn)?;
            let server = crate::mcp::McpServer::new(ctx.backend, ctx.ns, allow_query);
            eprintln!("kleos MCP {} on stdio", crate::mcp::MODERN);
            crate::mcp::serve_stdio(&server).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Attachment paths are made absolute here (a remote server can read them
/// only when it shares this filesystem; use the MCP `data_b64` form otherwise).
fn abs(p: &str) -> String {
    std::fs::canonicalize(crate::config::expand(p)).map(|x| x.to_string_lossy().into_owned()).unwrap_or_else(|_| p.to_string())
}

/// Prints a CLI error the way the output mode expects and returns the exit code.
pub fn report_error(json_mode: bool, err: &str) -> i32 {
    if json_mode {
        println!("{}", json!({"ok": false, "error": err}));
    } else {
        eprintln!("error: {err}");
    }
    1
}

impl CliCmd {
    /// Whether --json was asked for (to format errors).
    pub fn json_mode(&self) -> bool {
        fn j(c: &ConnArgs) -> bool {
            c.json
        }
        match self {
            CliCmd::Remember { conn, .. }
            | CliCmd::Recall { conn, .. }
            | CliCmd::Forget { conn, .. }
            | CliCmd::Reflect { conn }
            | CliCmd::Profile { conn, .. }
            | CliCmd::Query { conn, .. }
            | CliCmd::Mcp { conn, .. } => j(conn),
            CliCmd::Doc { cmd } => match cmd {
                DocCmd::Add { conn, .. } | DocCmd::Status { conn, .. } | DocCmd::List { conn } => j(conn),
            },
            CliCmd::Ontology { cmd } => match cmd {
                OntologyCmd::List { conn } | OntologyCmd::Learn { conn } | OntologyCmd::History { conn } | OntologyCmd::Undo { conn, .. } => j(conn),
            },
            CliCmd::Flow { cmd } => match cmd {
                FlowCmd::Start { conn, .. } | FlowCmd::Status { conn, .. } | FlowCmd::Signal { conn, .. } | FlowCmd::Cancel { conn, .. } | FlowCmd::List { conn, .. } => j(conn),
            },
            CliCmd::Branch { cmd } => match cmd {
                BranchCmd::Fork { conn, .. }
                | BranchCmd::Diff { conn, .. }
                | BranchCmd::Merge { conn, .. }
                | BranchCmd::Rewind { conn, .. }
                | BranchCmd::Discard { conn, .. }
                | BranchCmd::List { conn } => j(conn),
            },
        }
    }
}
