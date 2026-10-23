# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""A Python port of kleos's rule filter, `memory_claims::is_declarative`
(kleos-core/src/memory_claims.rs), for scoring single claims: the Python API
reports `decided` only for statements it split into several claims.
Kept line-for-line close to the Rust; the unit check at the bottom runs it on
the cases the Rust tests use.
"""

ABBR_NEVER = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "ft", "vs", "cf", "al", "approx",
    "appt", "apt", "dept", "est", "fig", "no", "nos", "vol", "vols", "pp", "ave", "blvd", "rd",
    "capt", "col", "lt", "sgt", "gen", "gov", "sen", "rep", "rev", "hon", "messrs", "mme", "mlle",
    "e.g", "i.e", "eg", "ie", "viz", "ca",
}
ABBR_MAYBE = {
    "etc", "inc", "ltd", "co", "corp", "llc", "plc", "bros", "a.m", "p.m", "am", "pm", "u.s",
    "u.k", "u.s.a", "ph.d", "phd", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    "min", "max", "hr", "hrs", "sec", "lb", "lbs", "oz", "km", "mi", "kg", "approx",
}
CLOSERS = set("\"'”’)]»*_")
OPENERS = "\"'“‘([«*_"
TERMINAL = set(".!?…。！？")

REQUEST_STARTS = [
    "please", "can you", "could you", "would you", "will you", "can u", "do you", "did you", "are you", "have you",
    "is there", "are there", "is it", "should i", "can i", "could i", "may i", "shall i", "would it", "help me",
    "tell me", "give me", "show me", "let me know", "remind me", "i need help", "i need some help", "i need advice",
    "i need some advice", "i need suggestions", "i need some suggestions", "i need ideas", "i need some ideas",
    "i need recommendations", "i need some recommendations", "i need tips", "i need some tips", "i'd like you to",
    "i would like you to", "i want you to", "i was wondering", "i'm wondering", "i am wondering",
    "i'm looking for suggestions", "i'm looking for recommendations", "i'm looking for ideas",
    "i'm looking for tips", "i'm looking for advice", "i'm looking for some", "any suggestions", "any tips",
    "any ideas", "any recommendations", "any advice", "suggest", "recommend", "list", "explain", "write",
    "provide", "find", "create", "generate", "describe", "compare", "summarize", "summarise", "draft", "plan",
    "brainstorm", "what", "how", "why", "which", "who", "whom", "whose",
]
WH_STARTS = {"what", "how", "why", "which", "who", "whom", "whose"}
COURTESY = ["thanks", "thank you", "thx", "ok", "okay", "great", "sounds good", "hi", "hello", "hey", "cool",
            "awesome", "perfect", "got it", "nice"]


def strip_bullet(line):
    t = line.lstrip()
    for b in ("- ", "* ", "• ", "– ", "— ", "+ ", "· "):
        if t.startswith(b):
            return t[len(b):]
    n = len(t) - len(t.lstrip("0123456789"))
    rest = t[n:]
    if 1 <= n <= 3:
        for m in (". ", ") "):
            if rest.startswith(m):
                return rest[len(m):]
    if len(t) >= 3 and t[0].isascii() and t[0].isalpha() and t[1] == ")" and t[2] == " ":
        return t[3:]
    return None


def blocks(text):
    out, opn = [], False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            opn = False
            continue
        item = strip_bullet(line)
        if item is not None:
            out.append(item.strip())
            opn = True
            continue
        cont = opn and line[0].islower() and out and out[-1] and not (out[-1][-1] in TERMINAL or out[-1][-1] == ":" or out[-1][-1] in CLOSERS)
        if cont:
            out[-1] += " " + line
        else:
            out.append(line)
        opn = True
    return out


def word_before(s, end):
    head = s[:end]
    i = max(head.rfind(" "), head.rfind("\t"), head.rfind("\n"))
    return head[i + 1:].lstrip(OPENERS)


def sentences(block):
    n = len(block)
    out, start, i = [], 0, 0
    while i < n:
        if block[i] not in TERMINAL:
            i += 1
            continue
        j = i
        while j < n and block[j] in TERMINAL:
            j += 1
        k = j
        while k < n and block[k] in CLOSERS:
            k += 1
        if k >= n or not block[k].isspace():
            i = max(j, i + 1)
            continue
        m = k
        while m < n and block[m].isspace():
            m += 1
        if m >= n:
            break
        q = m
        while q < n and block[q] in OPENERS:
            q += 1
        nxt = block[q] if q < n else " "
        upper, digit = nxt.isupper(), nxt.isascii() and nxt.isdigit()
        run = block[i:j]
        if run == ".":
            w = word_before(block, i)
            wl = w.lower()
            single = len(w) == 1 and w.isalpha()
            if wl in ABBR_NEVER:
                split = False
            elif single:
                split = upper and w.islower()
            elif wl in ABBR_MAYBE or ("." in wl and len(wl) <= 6):
                split = upper
            else:
                split = upper or digit
        elif "…" in run or run.startswith(".."):
            split = upper
        else:
            split = upper or digit
        if split:
            out.append(block[start:k].strip())
            start, i = m, m
        else:
            i = max(k, i + 1)
    tail = block[start:].strip()
    if tail:
        out.append(tail)
    return out


def words(s):
    out, cur = [], []
    for c in s:
        if c.isalnum():
            cur.append(c)
        elif cur:
            out.append("".join(cur).lower())
            cur = []
    if cur:
        out.append("".join(cur).lower())
    return out


def speaker_prefix(text):
    t = text.lstrip()
    i = t.find(": ")
    if i < 0:
        return None
    label = t[:i]
    ws = label.split(" ")
    if not label or len(ws) > 3 or len(label) > 40:
        return None
    if all(w and w[0].isupper() and all(c.isalnum() or c in ".'-’" for c in w) for w in ws):
        return t[i + 2:].lstrip()
    return None


def _sentence_is_declarative(claim):
    t = claim.strip().rstrip("".join(CLOSERS)).rstrip()
    if t.endswith("?") or t.endswith("？"):
        return False
    ws = words(t)
    if len(ws) < 3:
        return False
    low = t.lstrip(OPENERS).lower().replace("’", "'")

    def starts(p):
        return low.startswith(p) and (len(low) == len(p) or not low[len(p)].isalnum())
    for p in REQUEST_STARTS:
        if starts(p):
            if p in WH_STARTS and "," in low:
                continue
            return False
    if len(ws) <= 5 and any(starts(p) for p in COURTESY):
        return False
    return True


def is_declarative(claim):
    rest = speaker_prefix(claim)
    claim = rest if rest is not None else claim
    parts = [s for b in blocks(claim) for s in sentences(b)]
    if len(parts) > 1:
        return any(_sentence_is_declarative(p) for p in parts)
    return _sentence_is_declarative(claim)


if __name__ == "__main__":
    assert is_declarative("I live in Berlin.")
    assert not is_declarative("Can you recommend a hotel?")
    assert not is_declarative("Thanks, that helps!")
    assert not is_declarative("What's up with that incident? All good now?")
    assert is_declarative("When I was young, I lived in Paris.")
    assert not is_declarative("Suggest some books about Rome.")
    assert is_declarative("Thanks! I moved to Lisbon last May.")
    print("ok")
