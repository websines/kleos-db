# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Client for kleos's Postgres connector (docs/specs/pg-connector.md).

The connector runs in ``kleos server --pg-cdc <config.yaml>``; this module
calls its HTTP ops (``POST /pg/:op``) and writes connector configs.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Iterable, Optional

__all__ = ["PgClient", "PgError", "config_yaml"]


class PgError(RuntimeError):
    """A failed connector op (the server's message)."""


class PgClient:
    """The Postgres connector ops of a kleos server."""

    def __init__(self, url: str, token: Optional[str] = None, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def call(self, op: str, **args: Any) -> Any:
        body = json.dumps({k: v for k, v in args.items() if v is not None}).encode()
        req = urllib.request.Request(f"{self.url}/pg/{op}", data=body, method="POST")
        req.add_header("content-type", "application/json")
        if self.token:
            req.add_header("authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:  # the server's JSON error
            try:
                out = json.loads(e.read() or b"{}")
            except ValueError:
                raise PgError(f"pg {op}: HTTP {e.code}") from e
        if not out.get("ok", False):
            raise PgError(out.get("message") or f"pg {op} failed")
        return out.get("result")

    def status(self, source: Optional[str] = None) -> Any:
        """Lag (bytes, seconds), slot state, last LSN, per-table sync state,
        alerts; every source when ``source`` is None."""
        return self.call("status", source=source)

    def alerts(self) -> list[str]:
        """Alerts of every source, prefixed with the source name."""
        out = []
        for s in self.status():
            out.extend(f"{s['source']}: {a}" for a in s.get("alerts", []))
        return out

    def resync(self, source: str, tables: Optional[Iterable[str]] = None) -> Any:
        return self.call("resync", source=source, tables=list(tables) if tables else None)

    def map(self, source: str, infer: bool = False) -> Any:
        return self.call("map", source=source, infer=infer)

    def approve(self, source: str, version: Optional[int] = None) -> Any:
        """Review mode: apply a proposed mapping version to the ontology."""
        return self.call("approve", source=source, version=version)

    def query(self, source: str, sql: str, limit: Optional[int] = None, explain: bool = False) -> Any:
        """Read-only SQL over ``pg.<schema>.<table>`` and ``kleos.facts`` /
        ``kleos.nodes`` / ``kleos.memories`` (server built with pg-query)."""
        return self.call("query", source=source, sql=sql, limit=limit, explain=explain or None)

    def init(self, source: str) -> Any:
        return self.call("init", source=source)

    def drop_slot(self, source: str) -> Any:
        return self.call("drop_slot", source=source)


def _q(s: str) -> str:
    return json.dumps(s)  # JSON strings are valid YAML scalars


def config_yaml(
    source: str,
    connection: str,
    include: Iterable[str] = ("public.*",),
    exclude: Iterable[str] = (),
    ns: str = "default",
    writeback_allow: Iterable[str] = (),
    review: bool = False,
) -> str:
    """A connector config (YAML) with the common fields; everything else
    keeps its default (docs/specs/pg-connector.md, "Configuration")."""
    lines = [
        f"source: {source}",
        f"connection: {_q(connection)}",
        f"include: [{', '.join(_q(x) for x in include)}]",
    ]
    ex = list(exclude)
    if ex:
        lines.append(f"exclude: [{', '.join(_q(x) for x in ex)}]")
    lines.append(f"ns: {_q(ns)}")
    if review:
        lines.append("ontology: {review: true}")
    wb = list(writeback_allow)
    if wb:
        lines.append(f"writeback: {{allow: [{', '.join(_q(x) for x in wb)}]}}")
    return "\n".join(lines) + "\n"
