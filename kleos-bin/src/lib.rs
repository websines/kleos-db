/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Library side of the standalone `kleos` binary (kleos fork): the CLI, the
//! stateless MCP server and the backend they share
//! (docs/specs/mcp-cli-skills.md).

pub mod backend;
pub mod cli;
pub mod config;
pub mod mcp;
