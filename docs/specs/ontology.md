# Ontologies (`::ontology`)

*2026-10-24, branch `wave5/ontology`. Status: the parts marked **built** are on
the branch with tests and measurements (see "Results"); the rest is design.*

An ontology tells kleos what kinds of things and statements exist in a
namespace, and their rules: which classes there are, which properties and
relations connect them, how many values a property holds at a time, which
relations are symmetric, transitive or inverse of each other. kleos stores
ontologies as ordinary relations with valid time, uses the active one to type
what `remember` stores, and lets the rules drive three things:

1. **Supersession** (built): a new statement may replace an old one only when
   both are about the same subject and the same *single-valued* property.
   Small talk never takes part; a multi-valued property (likes, visited) never
   supersedes.
2. **Inference** (built): the ontology's characteristics (symmetric, inverse,
   transitive, sub-property, property chains) compile to a Datalog program over
   the stored facts, so "who are the user's relatives?" or "who is Anna's aunt?"
   is answered without anyone saying so.
3. **Validation** (built): facts are checked against domain, range, datatype
   and cardinality, and rejected or flagged.

The personal-memory ontology (`personal`) is only the default *pack*. Domain
ontologies are first-class: a trading desk imports its own (an OWL/RDF subset
in Turtle, or JSON), activates it on its namespaces, and kleos types,
supersedes, infers and validates with it. Nothing in kleos hardcodes the
personal labels: the extractor's labels and the claim typer's properties come
from the active ontology.

Why now: the first decider-on LoCoMo run (wave5/gliner-e2e, config B) made 14
automatic replacements in conversation 26, 13 of them wrong and all small talk
("Looks like you had a great day!" replaced by "I had a not-so-great experience
on a hike"), and left 442 pending decisions over 419 turns. kleos didn't know
which statements *can* replace each other. Cardinality is that knowledge.

## Model

### Terms

| Term | Meaning |
|---|---|
| **ontology** | A named, versioned set of classes and properties (`onto`, e.g. `personal`, `acme-trading`) |
| **class** | A kind of node: Person, Place, Organization, Trade, Instrument. Classes form a DAG (`parents`) |
| **property** | A typed statement about a node. `kind` is `object` (node → node), `data` (node → literal) or `role` (event node → participant or literal; see N-ary facts) |
| **cardinality** | `one` (at most one current value per subject: lives_in, employer, age, follower_count) or `many` (likes, visited, owns, sibling_of) |
| **characteristics** | `symmetric`, `transitive`, `inverse: <prop>`, `parents` (sub-property of), `chain: [p1, p2]` (p1 then p2 implies this) |
| **temporal** | The value changes over time (lives_in, job_title, follower_count: true) or not (birthday, born_in: false). Non-temporal single-valued properties never auto-supersede: a different value is a conflict for review, not an update |
| **node** | Anything facts are about: a memory-API entity, the speaker (`user`), a relative known by relation (`user.sister`), an event node (`ev:…`), or a literal-valued subject. Identified by a key string; see "Nodes" |
| **fact** | `(subject, property, object)` with valid time, a confidence, who decided it and the memory it came from |
| **claim type** | What `remember`'s typer concluded about one claim: its class (`fact`, `smalltalk`, `untyped`), subject, property and score |

### Storage (built)

Ontology definitions are shared by every namespace of the store; activation and
facts are per namespace. Every definition relation has valid time
(`vt: Validity`), so an ontology evolves without losing what it said before, and
`@` reads see the ontology as of any time. Redefining a class or property
asserts a new version; removing one appends a retraction.

```
kleos_onto        {onto, vt: Validity => version: String, doc: String, source: String}
kleos_onto_class  {onto, name, vt: Validity => parents: [String], label: String, doc: String, aliases: [String]}
kleos_onto_prop   {onto, name, vt: Validity => kind: String, domain: [String], range: [String],
                   datatype: String?, card: String, symmetric: Bool, transitive: Bool, inverse: String?,
                   parents: [String], chain: [String], temporal: Bool, unit: String?,
                   label: String, doc: String, aliases: [String], cues: [String]}
kleos_onto_ns     {ns, vt: Validity => ontos: [String], policy: String, untyped: String}
kleos_onto_fact   {ns, s, p, o, vt: Validity => lit: Json?, conf: Float, by: String, mem: String?, status: String}
kleos_onto_node   {ns, node, vt: Validity => class: String, label: String?, mem: String?}
kleos_mem_type    {ns, id => onto: String, cls: String, subject: String?, prop: String?, score: Float, by: String}
```

- `kleos_onto_ns` activates a *list* of ontologies on a namespace, later ones
  overriding earlier ones by name: `["personal", "acme-trading"]` extends the
  default, `["acme-trading"]` replaces it. `policy` is the supersession policy
  (`off`, `cardinality`), `untyped` what happens when a claim couldn't be typed
  (see "Supersession"). With no row, the namespace has no ontology and
  `remember` behaves exactly as before.
- `kleos_onto_fact`: `o` is a node key, or for a literal the literal's canonical
  string (`"1500 EUR"`) with the typed value in `lit`
  (`{"value": 1500, "datatype": "xsd:decimal", "unit": "EUR"}`). Keying by `o`
  lets multi-valued facts coexist. Asserting a value of a single-valued
  property retracts the subject's other current value in the same transaction
  (cardinality enforced on write).
- `status` is `current` or `pending`. Who may write `current` depends on how
  reliable the source is (docs/research/gliner-usage.md, "Relations"):
  - asserted by the app (`::ontology assert`), or first-person kinship and role
    edges by rule ("my sister Anna" → `sister_of(anna, user)`): `current`;
  - general relations extracted from free text (works_at, lives_in from GLiNER2
    relation extraction, relation F1 0.18–0.37 at this size): `pending`, promoted
    to `current` only when the decider confirms the pair or the same edge recurs
    in two or more statements.

  Inference and validation read `current` facts by default (`status: 'any'`
  includes pending).
- `kleos_mem_type` holds one row per typed claim, written in `remember`'s own
  transaction. Memory ids are immutable, so it has no valid time. Rows are a
  cache: a memory without one is typed on the fly from its text and vector
  (stores that predate the ontology, or a memory typed under an older version).
- Branches (docs/specs/memory-branches.md): ontology activation and facts read
  through the lineage like memories do. Not built yet; a fork today starts with
  no activation row and inherits nothing (noted in "Left").

### Nodes

A node is identified by a **key**:

| Key | Node |
|---|---|
| `user` | the person speaking in a single-user store |
| a normalised name (`anna`, `spotify`) | a named entity; matches the memory API's entity of that normalised name (`kleos_mem_entity.norm`) when there is one |
| `<owner>.<relation>` (`user.sister`, `caroline.mother`) | a person known only by their relation to another node; merged into a named node when a name arrives ("my sister Anna" gives both keys and the fact `sister_of(anna, user)`) |
| `ev:<memory id>[.<n>]` | an event node (N-ary facts) |
| `lit:<canonical>` | never a subject; literals live in `o` + `lit` |

Kinship edges are multi-valued and never superseded (two sisters are both
true); role relations are time-varying, and employer and partner are
single-valued, so a new employer supersedes the old one under the cardinality
rule. The GLiNER port (`wave5/gliner-port`) emits mentions and edges in this
shape: a person node with a lookup key plus a typed edge, possessor resolved,
pronouns inheriting the most recent mention of the statement.

Names are keys rather than entity ids so that facts survive entity merges and
work in stores that don't use entity resolution; `::ontology` reads map a key
to its canonical memory entity when one exists. This follows the GLiNER research
recommendation (docs/research/gliner-usage.md, a6–a7): a relative is a node
with a lookup key plus a typed edge, the possessor is resolved (first person →
`user` or the speaker; "his wife" → the most recent person of the statement),
and a later name is merged into that node, not given a second one.

### N-ary facts and events (built: storage and assert; design: extraction)

A trade has an instrument, a quantity, a price, a counterparty, a desk and a
time. Binary facts can't hold that without losing which quantity goes with
which price, so kleos reifies it as an **event node** (neo-Davidsonian style,
the same pattern as RDF reification by an intermediate node, Wikidata
qualifiers and OWL n-ary relation pattern 1):

```
node  ev:m_123   class Trade
fact  (ev:m_123, instrument,   aapl)
fact  (ev:m_123, quantity,     "500")        lit {"value": 500, "datatype": "xsd:integer"}
fact  (ev:m_123, price,        "187.20 USD") lit {"value": 187.2, "datatype": "xsd:decimal", "unit": "USD"}
fact  (ev:m_123, counterparty, goldman_sachs)
fact  (ev:m_123, desk,         equities_ny)
fact  (ev:m_123, traded_at,    "2026-10-24T14:31:00Z") lit {"datatype": "xsd:dateTime", ...}
```

- An event class is any class under `Event`; its properties have
  `kind: 'role'`, `domain: [<event class>]`. Role cardinality applies *per
  event* (one price per trade), so an event's roles never supersede another
  event's: two trades are two facts, never an update. That is exactly the
  "multi-valued at the subject level, single-valued at the event level" rule a
  memory system needs for transactions, visits, meals and meetings.
- `::ontology assert {facts: [...], event: {class, roles}}` writes an event
  node and its role facts in one transaction (built). Apps that already have
  structured records (a trading system's fills) assert them directly; free text
  is typed by the extractor (below).
- **Typed literals:** `lit.datatype` is an XSD name (`xsd:string`,
  `xsd:integer`, `xsd:decimal`, `xsd:boolean`, `xsd:date`, `xsd:dateTime`,
  `xsd:duration`), plus `unit` (a UCUM code such as `kg`, `h`, `km`) or a
  currency (ISO 4217 code: `USD`, `EUR`) for amounts. Validation checks a fact
  against the property's `datatype` and `unit`. Unit *conversion* is not built.

### Default pack: `personal` (built)

52 properties over 9 classes (Person, Place, Organization, Pet, Thing, Event,
Activity, Food, Media), seeded from the 44 closed attributes of the GLiNER key
experiment (`scripts/gliner/ft_keys.py`) and the life-stream facts. The pack is
JSON compiled into kleos (`kleos-core/src/ontology_personal.json`), loaded on
first use. Examples:

| Property | Kind | Card | Temporal | Characteristics |
|---|---|---|---|---|
| lives_in | object Person → Place | one | yes | |
| employer | object Person → Organization | one | yes | |
| job_title | data | one | yes | |
| partner | object Person → Person | one | yes | symmetric |
| sibling_of | object Person → Person | many | no | symmetric |
| sister_of | object Person → Person | many | no | parents [sibling_of] |
| parent_of | object Person → Person | many | no | inverse child_of |
| aunt_of | object Person → Person | many | no | chain [sister_of, parent_of] |
| pet | object Person → Pet | many | yes | |
| likes / dislikes | data or object | many | yes | |
| visited | object Person → Place | many | no | |
| favourite_food | data | one | yes | |
| age, birthday | data | one | age yes, birthday no | |
| follower_count, savings, rent | data, number with unit | one | yes | |
| located_in | object Place → Place | many | no | transitive |

## Claim typing (built: rules + embedding prototypes; hook for extractors)

`remember` types each declarative claim against the namespace's active
ontology. A claim type is `(cls, subject, property, score, by)`:

1. **Small talk** (`cls: smalltalk`): reactions, greetings, thanks, compliments
   and sympathy addressed to the other speaker ("Wow, Caroline! Sorry that
   happened to you.", "Looks like you had a great day!", "Hey Mel, long time no
   chat!"), and photo-only turns. Rules in `ontology_type.rs`: an interjection
   or reaction opener, or second person without first person, and no fact cue.
   Small talk is stored and recallable but never decided.
2. **Cues** (`by: cue`): each property may list cue phrases (`lives_in`: "live
   in", "moved to", "my apartment in"…), matched at word boundaries. The
   longest match wins. Domain packs carry their own cues; imported ontologies
   get their labels and aliases as cues.
3. **Embedding prototypes** (`by: proto`): when the store has an embedder, each
   property's `label: doc (aliases)` is embedded once per (ontology version,
   embedder) and cached. A claim takes the nearest property if its cosine is at
   least `0.45` and beats the runner-up by `0.02`; otherwise it is `untyped`.
   The claim's vector is the one `remember` already computed, so typing costs
   one dot product per property.
4. **Subject:** the speaker (`Name:` prefix) or `user` for first-person claims;
   `<speaker|user>.<kin>` for "my sister …"; a leading proper name as a guess,
   marked `name:anna` (no NER here: "Words just have a magical way…" would
   otherwise be a person), which never decides a block; otherwise unknown.
   Discourse openers, interjections, dates and vocatives ("Anna, …") are not
   names. A claim whose subject is a relative is typed by the relative's
   property ("My sister Anna lives in Munich." is `lives_in` of
   `user.sister`), never by the relation itself.
5. **Text features:** the speaker label, the content words (speaker label and
   photo captions left out), and whether the claim names something concrete
   (a number, or an inner capitalised word that isn't a vocative).
6. **Extractor hook:** an extractor (the GLiNER port) can replace or confirm
   steps 2–4.

Labels come from the active ontology, never from code: the cues and
prototypes are the ontology's own, and an open-label extractor like GLiNER
takes the active ontology's classes and relations, with their descriptions,
as its labels. The kleos GLiNER fine-tune (Mazino0/kleos-gliner) is
personal-domain and trained on four bare labels, so it may have lost general
ability (docs/research/gliner-usage.md: mix 30–50% general NER data or train a
LoRA adapter). Domain ontologies therefore use upstream GLiNER2.5 with label
descriptions; the fine-tune is for the `personal` pack. Wiring the extractor's
labels to the active ontology is not built yet (see "Left"). GLiNER is open-label at inference, so the
   active ontology's classes and relations, with their descriptions, become its
   labels. Trade-off: the kleos fine-tune is personal-domain and was trained on
   four bare labels; it may have lost general ability (the research found no
   evidence either way and recommends mixing 30–50% general NER or a LoRA
   adapter). So domain ontologies use **upstream GLiNER2.5** with label
   descriptions as the fallback, and the fine-tune only for the `personal` pack.

## Supersession (built)

`kleos_onto_ns.policy`:

- `off`: exactly today's behaviour.
- `cardinality` (default once an ontology is activated). For a new claim *c*
  and a candidate *d*:

| Case | Supersede | Duplicate |
|---|---|---|
| *c* or *d* is small talk | not asked | not asked |
| subjects certainly different (two speakers' first persons, a speaker and their own relative, two relatives) | not asked | not asked |
| different speaker labels ("Caroline:" vs "Melanie:") | not asked | not asked |
| both typed, same property, `card: one`, `temporal: true` | asked (normal bands) | asked |
| both typed, same property, `card: one`, `temporal: false` | capped at pending (a conflict to review) | asked |
| both confidently typed, same `card: many` property, or different properties | not asked | asked |
| otherwise, with a side untyped or typed without confidence | per `untyped` | asked |

A type is *confident* when its prototype score is at least 0.55, or 0.50 with a
cue phrase, or it came from cues alone (no embedder). A weak type counts as
untyped for the gate, because typing is right often enough to help but not
always (two claims about bikes were typed `vehicle` and `commute`).

`untyped` is `decide` (today's decision for that pair), `pending` (asked, but
an auto answer is capped at pending) or `dup_only` (not asked). Under `decide`,
a **thin** pair is still capped at pending: a side with nothing concrete and
fewer than five content words ("Nah, I haven't.", "It was super peaceful and
refreshing."). Short concrete facts ("I play tennis on Saturdays.", "I drive a
2015 Volkswagen Golf.") are not thin.

**The default is `decide`.** `pending` cut the remaining wrong LoCoMo
replacements further but dropped knowledge-update latest-first from 83% to 70%
(91% either way with `hide_pending`), because many true updates are long chatty
claims the typer leaves untyped. See "Results".

The thresholds (`PROTO_MIN` 0.45, `PROTO_MARGIN` 0.02, `CUE_BONUS` 0.06,
`CONFIDENT` 0.55, `THIN` 5) can be overridden for evaluation with
`KLEOS_ONTO_<NAME>` environment variables.

Event nodes and roles (N-ary facts) never supersede across events.

## Inference (built)

`::ontology infer {ns, at?, node?, prop?}` compiles the active ontology into a
CozoScript program over `kleos_onto_fact` (at `at`, default now) and returns the
asserted and derived facts, each with `derived: true|false`:

```
f[s, p, o]      := *kleos_onto_fact{ns: $ns, s, p, o @ $at}
f[o, p, s]      := f[s, p, o], p in $symmetric
f[o, q, s]      := f[s, p, o], inv[p, q]               # inverse, both directions
f[s, q, o]      := f[s, p, o], sup[p, q]               # sub-property
f[s, p, o2]     := f[s, p, o], f[o, p, o2], p in $transitive
f[s, r, o2]     := f[s, p, o], f[o, q, o2], chain[r, p, q]
```

Recursion is semi-naive, so transitive closures and chains of chains terminate.
The program is generated per call from whatever ontology is active, so it works
for imported ontologies (a FIBO-derived `isSubsidiaryOf` declared transitive
gets its closure) exactly like for the default. `recall` can optionally expand
over inferred edges (design; not built).

## Validation (built)

`::ontology validate {ns}` and `assert {..., validate: 'reject'|'flag'|'off'}`:

- the property exists in the active ontology (or an alias of one);
- domain and range: the subject's and object's classes (`kleos_onto_node`,
  through the class DAG) must be under one of the listed classes; unknown class
  passes with a warning;
- data properties: `lit.datatype` must match `datatype`; `unit` must match when
  the property has one;
- cardinality: more than one current value of a `card: one` property for a
  subject;
- `reject` refuses the whole assert; `flag` writes and returns the violations.

## Ops (built)

`::ontology <op> {opts}`, `Db::ontology_call(op, json)`,
`CozoDbPy.ontology_call(op, dict)`, `POST /ontology/:op`:

| Op | Does |
|---|---|
| `define {onto, classes?, properties?, doc?, version?}` | create or update (new valid-time version) |
| `import {onto, format: 'json'|'turtle', text}` | parse and define |
| `export {onto, format: 'json'|'turtle'}` | the active version |
| `get {onto}`, `list` | read |
| `activate {ns, ontos, policy?, untyped?}`, `deactivate {ns}` | per namespace |
| `assert {ns, facts: [{s, p, o, lit?}], event?, validate?}`, `retract {ns, facts}` | facts |
| `facts {ns, s?, p?, o?, at?}`, `infer {...}`, `validate {ns}` | read, derive, check |
| `type {ns, text}` | what the typer would conclude for a text (debugging) |

## Import and export

### JSON (built)

```json
{"onto": "acme-trading", "version": "2026-10", "doc": "...",
 "classes": [{"name": "Trade", "parents": ["Event"], "label": "trade", "doc": "...", "aliases": ["fill", "execution"]}],
 "properties": [{"name": "price", "kind": "role", "domain": ["Trade"], "datatype": "xsd:decimal",
                 "card": "one", "unit": "currency", "doc": "execution price", "aliases": ["px", "fill price"]}]}
```

YAML is the same document; kleos takes JSON, and a YAML file converts with any
YAML-to-JSON tool (no YAML parser is vendored).

### OWL/RDF subset in Turtle (built)

A hand-written Turtle reader for the subset that ontologies use. Supported:
`@prefix`/`PREFIX`, `@base`, prefixed names and `<IRI>`s, `a`, `;` and `,`
lists, string literals with language tags or datatypes, numbers, booleans,
comments. Mapping:

| Turtle | kleos |
|---|---|
| `owl:Class`, `rdfs:Class` | class |
| `rdfs:subClassOf X` (named X) | parent |
| `owl:ObjectProperty` | `kind: object` |
| `owl:DatatypeProperty` | `kind: data` |
| `owl:FunctionalProperty` | `card: one` |
| `owl:SymmetricProperty`, `owl:TransitiveProperty` | flags |
| `owl:inverseOf` | `inverse` (both directions) |
| `rdfs:subPropertyOf` | `parents` |
| `owl:propertyChainAxiom ( p q )` | `chain` |
| `rdfs:domain`, `rdfs:range` (named classes or `xsd:*`) | domain, range / datatype |
| `rdfs:label`, `skos:prefLabel` | label (English preferred) |
| `skos:altLabel`, `skos:hiddenLabel` | aliases (→ cues) |
| `rdfs:comment`, `skos:definition` | doc |

Ignored with a warning count: restrictions (`owl:Restriction` blank nodes),
unions, individuals, imports, annotations other than the above. Local names
(the part after `#` or the last `/`) become kleos names; the full IRI is kept in
`doc` provenance. Export writes the same subset back.

**Reference example, finance: FIBO** (EDM Council's Financial Industry Business
Ontology), MIT License, © 2020 Enterprise Data Management Council (checked
2026-10-24 at github.com/edmcouncil/fibo/blob/master/LICENSE). Not vendored.
FIBO's modules are RDF/XML; convert a module to Turtle (any RDF tool) and
`import` it. Most of FIBO's semantics are in restrictions, which this subset
skips, so an imported FIBO module gives classes, the property hierarchy,
domains, ranges, inverses and labels, but not cardinality restrictions:
mark the single-valued properties a desk cares about with a small JSON overlay
activated after it (`["fibo-fnd", "acme-overrides"]`).

## Ontology learning (design only; follow-up)

Propose classes and properties from a namespace's memories, with a human
approving them before they become active:

1. Cluster untyped declarative claims by embedding (HDBSCAN or k-means over the
   claim vectors kleos already has).
2. For each cluster, propose a property: a label from the cluster's frequent
   head nouns and verbs (or an LLM call if one is configured), a cardinality
   from the data (do subjects ever hold two current values at once? do new
   values arrive with update cues like "now", "anymore"?), and cues from the
   n-grams that separate the cluster.
3. Write proposals to `kleos_onto_proposal {ns, id => kind, body: Json, evidence:
   [mem ids], status}`; `::ontology proposals`, `approve {id}`, `reject {id}`.
   Approval defines them in a namespace-local ontology layered on top.
4. The decider can score a proposal's cardinality by asking `supersedes` on
   sampled same-cluster pairs.

Not built: the clustering and proposal ops. The `untyped` share per namespace
(`::ontology stats`) is the signal for when it's worth running.

## Migration

- Schema: the ontology relations are created on first use of `::ontology`
  (or of a namespace with an activation row). No memory schema version bump;
  stores without them behave exactly as before.
- Off by default: a namespace without `activate` is untouched. `memory_init
  {ontology: 'personal'}` activates the default pack for the store's default
  namespace (built).

## Results

*2026-10-24, wave5/ontology on main beb1bcd. Real models: EmbeddingGemma int8
and decider-nano on Metal through the Python wheel. Scripts in
`scripts/ontology/`; data in `.bench-data/ontology/`.*

### Knowledge-update questions (the 23 held out from decider-nano training)

`scripts/memory-api/claims_ku.py held` with `KLEOS_ONTOLOGY`, decider on:

| | latest first | with `hide_pending` | auto supersedes / q | pending / q |
|---|---|---|---|---|
| no ontology (gliner-e2e baseline) | 83% | 91% | 0.43 | 1.83 |
| `personal`, `untyped: decide` (default) | **83%** | **91%** | 0.43 | 1.74 |
| `personal`, `untyped: pending` | 70% | 91% | 0.22 | 1.96 |

The policy kept 2.3 candidate pairs per question out of `supersedes` and none
of them was a true update. All 22 hand-checked true update pairs
(`.bench-data/gliner/ku_pairs.json`) pass the policy (`allow`); both claims got
the same property in 10 of 22, and the rest are long chatty claims the typer
leaves untyped (20 of 44 claims), which `decide` passes through.

### Life stream (40 turns, scripted moves, job changes, preferences)

`scripts/ontology/life_stream_py.py`, decider on:

| | auto supersedes right / wrong / missed | top-1 | top-1 `hide_pending` | pending |
|---|---|---|---|---|
| no ontology | 12 / 0 / 3 | 15/20 | 19/20 | 9 |
| `personal` | **12 / 0 / 3** | **15/20** | **19/20** | 8 |

Two bugs found on the way, both fixed: "My sister Anna moved from Munich to
Hamburg" was typed `sister_of` (a relation) and so never superseded "My sister
Anna lives in Munich" (claims about a relative are now typed by the relative's
property); and a first thin-claim rule capped "I quit Siemens and joined
Spotify" against "I work as a software engineer at Siemens" (short concrete
facts are now exempt).

### LoCoMo, decider on (the decider-on run of wave5/gliner-e2e, config B)

**Post hoc**, the policy applied to every `supersedes` decision of B's ten
conversations (`typing_eval.py`):

| B's action | blocked (no decision) | capped at pending | allowed |
|---|---|---|---|
| 180 automatic supersedes | 108 | 65 | 7 |
| 1,976 pending supersedes | 1,045 | 738 | 190 (3 `no_supersede`) |

Hand check: every blocked supersede read was a wrong replacement (reactions,
greetings, photo turns, two speakers); the seven still allowed are wrong too
(two different tournaments, a portrait and an aim, a park described twice).
So the policy lets no true update through on LoCoMo because there are almost
none to let through; what matters is that it removes the wrong ones.

**End to end**, the first three conversations (1,451 turns) ingested with the
ontology active (`retrieval.py --variant sup --ontology personal`), against B
on the same conversations:

| | auto supersedes | pending (supersedes / duplicates) | auto duplicates | any-evidence hit @5 / @10 |
|---|---|---|---|---|
| B (no ontology) | 42 | 1,663 (387 / 1,276) | 26 | 0.718 / 0.791 |
| `personal` | **1** | **693** (231 / 462) | 3 | 0.715 / 0.786 |

The one remaining auto supersede is wrong ("Here's a recent self-portrait I
made last week." over "My aim was to show warmth, love and self-acceptance.").
B's 26 auto duplicates were repeated greetings ("No worries, John."), which
small talk no longer decides. Retrieval is unchanged within a question or two.
Ingest took 1,071 s against B's 4,158 s, but B ran under heavier load, so only
the direction (fewer decider calls: small talk is never sent) is meaningful.

**QA on the same three conversations** (all 385 non-adversarial questions,
context output, GLM-4.7-Flash reader and judge, `qa.py --conditions
ctx:sup+default,ctx:sup+onto+default`):

| | all | single-hop (74) | temporal (90) | multi-hop (21) | open-domain (200) |
|---|---|---|---|---|---|
| B (decider on, no ontology) | 0.852 | 0.865 | 0.911 | 0.810 | 0.825 |
| `personal` | 0.847 | 0.851 | 0.900 | 0.762 | 0.830 |

Two questions apart, with 9 flips one way and 11 the other: no measurable
effect on answers. 154 of the 385 contexts were identical. The policy's value
on LoCoMo is a cleaner store (41 fewer wrong replacements, 970 fewer pending
decisions to review) at the same answer quality, not higher scores. About 1,085
reader and judge calls, roughly 0.65 points of the weekly plan.

### Claim typing, hand sample

40 random LoCoMo claims: 12 typed small talk (11 right; "They love running and
playing fetch, you should see them" is a fact about the dogs), 1 typed fact
(`attended` for a question about an event, wrong), 27 untyped (vague turns,
reasonably left untyped, though a few were typable: "prepping for my dance
studio", "stayed up all night playing games"). Typing recall on chat is low;
the policy is built so that an untyped claim falls back to today's decision.

## Left

- **Branches:** activation rows and facts don't read through a branch's
  lineage yet; a fork starts with no ontology.
- **Extractor labels from the ontology:** the GLiNER port (main beb1bcd) uses
  its four trained labels. Feeding the active ontology's classes and relations
  to it as labels, with upstream GLiNER for domain ontologies, is not wired.
- **Two kinship stores:** the GLiNER port writes relative edges to
  `kleos_mem_link` (`decided_by: 'gliner-rule'`); the ontology's rule writes
  `kleos_onto_fact`. They agree on shape (person node plus typed edge,
  possessor resolved); one should read the other.
- **Recall over inferred edges**, ontology learning, unit conversion: designed,
  not built.
- **Typing recall on chat is low** (most LoCoMo claims are untyped); the
  small-talk lexicon misses some courtesies ("No problem, John.").
