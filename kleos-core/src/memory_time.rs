/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Rule-based English temporal expressions for the memory API
//! (docs/specs/memory-api.md, "Event time"): every date, relative day,
//! offset, weekday, week, month, season, year, "since" / "until", duration
//! and range in a text, resolved against the time it was said (`anchor`, a
//! statement's valid-from) into an interval `[start, end)` of whole UTC days
//! with a granularity. No model; deterministic; a cheap pre-check skips texts
//! with no digit and no time word.
//!
//! **Resolution** (the anchor's UTC calendar; weeks start on Monday):
//! - `yesterday`, `today`, `tonight`, `tomorrow`, `the day before
//!   yesterday`, parts of day (`this morning`, `last night` = the day before,
//!   `yesterday afternoon`) → that day.
//! - `N days ago` → that day; `N weeks ago` → the week containing anchor −
//!   7N days; `N months ago` → that calendar month; `N years ago` → that
//!   calendar year. `in N days/weeks/months` and `N days from now` likewise
//!   forward (`in a day` and `in a year` are skipped: "twice in a day").
//! - `last Friday` → the latest Friday strictly before the anchor day; `next
//!   Friday` → the first strictly after; `this Friday` → the Friday of the
//!   anchor's week. A bare weekday (`on Friday`, `Friday`) is the latest one
//!   strictly before the anchor day, or the first strictly after it when its
//!   sentence has a future cue (will, 'll, going, gonna, plan, next,
//!   tomorrow, soon, …). `every/each/that Friday` and `Fridays` are skipped.
//! - Weekends are Saturday–Sunday (granularity day, two days): `this
//!   weekend` is the one of the anchor's week (the coming one on a weekday,
//!   the current one on a weekend); `last weekend` the latest one whose
//!   Sunday is before the anchor day; `next weekend` = this one + 7 days;
//!   `this past` / `over the weekend` the latest one whose Saturday is before
//!   the anchor day (`over the weekend` with a future cue: the coming one).
//! - `last/this/next week|month|year` are calendar units; `the year before
//!   last` / `after next` two away.
//! - Seasons are northern meteorological ones: spring Mar–May, summer
//!   Jun–Aug, fall Sep–Nov, winter Dec–Feb, named by the year it starts
//!   (`winter 2022` = Dec 2022 – Feb 2023). `last summer`: the previous
//!   year's when the anchor is in summer, else the latest summer that has
//!   ended. `this summer`: the current one when in it, else the anchor
//!   year's. `next summer`: the first one starting after the anchor (the
//!   anchor year's when said in spring). A season alone is not a date; it
//!   needs last/this/next or a year (`summer 2022`, `the fall of 2021`).
//! - A month without a year (`in March`, `early June`, `since March`,
//!   `until June`): the latest such month at or before the anchor month,
//!   or, with a future cue (or `until`), the first at or after it. `last
//!   March` is strictly before the anchor month, `next March` strictly after,
//!   `this March` in the anchor year.
//! - A month and day without a year (`May 7`, `the 7th of May`): the
//!   occurrence nearest the anchor (ties go to the past), or the first at or
//!   after the anchor day with a future cue.
//! - Numeric dates: `2023-05-07` / `2023/05/07`; `5/7/2023` and `05/07/23`
//!   are US month/day unless the first number is above 12. Two-digit years
//!   are 20xx below 70, else 19xx. Day–month numbers without a year (`9/11`)
//!   are skipped.
//! - `since X` → `[start of X, end of the anchor day)`; `until/till/through
//!   X` → `[anchor day, end of X)` (a past "until" keeps only X itself);
//!   `for the past N weeks`, `in the last 2 months`, `for 3 years now`, and
//!   `for N units` in a present-perfect sentence (have / has / 've / 's
//!   been) → `[anchor − N units, end of the anchor day)`; a plain `for two
//!   weeks` is a length, not a time, and is skipped.
//! - Ranges: `from X to Y`, `between X and Y`, `X – Y` of two absolute
//!   dates, `May 3-7`, `3–7 May 2023`, `2019-2021`, `from March to May` →
//!   `[start of X, end of Y)` with the coarser granularity.
//!
//! **Conservative by design** (a missed date is better than a wrong one):
//! month names must be capitalised and need a day, a year or a cue (`in`,
//! `since`, `last`, `early`, `end of`, …), so the modal "may", the verb
//! "march" and names like June or August alone never match; abbreviated
//! "Jan" needs a year or a period; seasons need last/this/next or a year; a
//! bare four-digit number is a year only after a cue (`in`, `since`,
//! `until`, `from`, `during`, `back in`, `mid-`) or inside a date; clock
//! times, ordinals ("the second time"), vague phrases ("a few days ago",
//! "the other day", "recently", "one day"), decades and numbers glued to
//! digits, slashes, currency or dashes (versions, prices, phone numbers,
//! scores, "24/7", "9/11", fractions) are ignored. When two matches overlap
//! the longer one wins.

use std::sync::LazyLock;

use chrono::{Datelike, Duration, Months, NaiveDate, TimeZone, Utc, Weekday};
use regex::{Captures, Regex};
use serde_derive::{Deserialize, Serialize};

const DAY_US: i64 = 86_400_000_000;

/// How precise a resolved time is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Granularity {
    /// One or a few days (also parts of day and weekends).
    Day,
    /// A Monday-start week.
    Week,
    /// A calendar month.
    Month,
    /// A meteorological season.
    Season,
    /// A calendar year.
    Year,
}

impl Granularity {
    /// `day`, `week`, `month`, `season` or `year`.
    pub fn as_str(self) -> &'static str {
        match self {
            Granularity::Day => "day",
            Granularity::Week => "week",
            Granularity::Month => "month",
            Granularity::Season => "season",
            Granularity::Year => "year",
        }
    }

    /// The inverse of [`Granularity::as_str`].
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "day" => Granularity::Day,
            "week" => Granularity::Week,
            "month" => Granularity::Month,
            "season" => Granularity::Season,
            "year" => Granularity::Year,
            _ => return None,
        })
    }
}

/// What kind of time expression matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeKind {
    /// A time: a day, week, month, season or year.
    Point,
    /// `since X`: from X to the anchor.
    Since,
    /// `until X`: from the anchor to the end of X.
    Until,
    /// `from X to Y`.
    Range,
    /// `for the past N weeks`: N units up to the anchor.
    Duration,
}

impl TimeKind {
    /// `point`, `since`, `until`, `range` or `duration`.
    pub fn as_str(self) -> &'static str {
        match self {
            TimeKind::Point => "point",
            TimeKind::Since => "since",
            TimeKind::Until => "until",
            TimeKind::Range => "range",
            TimeKind::Duration => "duration",
        }
    }

    /// The inverse of [`TimeKind::as_str`].
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "point" => TimeKind::Point,
            "since" => TimeKind::Since,
            "until" => TimeKind::Until,
            "range" => TimeKind::Range,
            "duration" => TimeKind::Duration,
            _ => return None,
        })
    }
}

/// One resolved temporal expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeExpr {
    /// Start (µs since the epoch, inclusive, UTC midnight).
    pub start: i64,
    /// End (µs, exclusive, UTC midnight).
    pub end: i64,
    /// Precision of the interval.
    pub granularity: Granularity,
    /// Point, since, until, range or duration.
    pub kind: TimeKind,
    /// The matched text as written.
    pub expr: String,
    /// Byte range of `expr` in the input.
    #[serde(skip)]
    pub span: (usize, usize),
}

// ------------------------------------------------------------------ calendar

fn date_of(us: i64) -> NaiveDate {
    Utc.timestamp_micros(us)
        .single()
        .map(|d| d.date_naive())
        .unwrap_or(NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
}

fn us_of(d: NaiveDate) -> i64 {
    d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp_micros()
}

fn days(d: NaiveDate, n: i64) -> NaiveDate {
    d.checked_add_signed(Duration::days(n)).unwrap_or(d)
}

fn add_ym(y: i32, m: u32, n: i32) -> (i32, u32) {
    let t = y * 12 + (m as i32 - 1) + n;
    (t.div_euclid(12), (t.rem_euclid(12) + 1) as u32)
}

fn month_iv(y: i32, m: u32) -> Option<(NaiveDate, NaiveDate)> {
    let (ny, nm) = add_ym(y, m, 1);
    Some((NaiveDate::from_ymd_opt(y, m, 1)?, NaiveDate::from_ymd_opt(ny, nm, 1)?))
}

fn year_iv(y: i32) -> Option<(NaiveDate, NaiveDate)> {
    Some((NaiveDate::from_ymd_opt(y, 1, 1)?, NaiveDate::from_ymd_opt(y + 1, 1, 1)?))
}

fn week_start(d: NaiveDate) -> NaiveDate {
    days(d, -(d.weekday().num_days_from_monday() as i64))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Season {
    Spring,
    Summer,
    Fall,
    Winter,
}

fn season_name(s: &str) -> Season {
    match s.to_ascii_lowercase().as_str() {
        "spring" => Season::Spring,
        "summer" => Season::Summer,
        "winter" => Season::Winter,
        _ => Season::Fall,
    }
}

/// The season named by the year it starts in.
fn season_iv(s: Season, y: i32) -> Option<(NaiveDate, NaiveDate)> {
    let (m0, m1, y1) = match s {
        Season::Spring => (3, 6, y),
        Season::Summer => (6, 9, y),
        Season::Fall => (9, 12, y),
        Season::Winter => (12, 3, y + 1),
    };
    Some((NaiveDate::from_ymd_opt(y, m0, 1)?, NaiveDate::from_ymd_opt(y1, m1, 1)?))
}

fn season_of(d: NaiveDate) -> (Season, i32) {
    match d.month() {
        3..=5 => (Season::Spring, d.year()),
        6..=8 => (Season::Summer, d.year()),
        9..=11 => (Season::Fall, d.year()),
        12 => (Season::Winter, d.year()),
        _ => (Season::Winter, d.year() - 1),
    }
}

fn month_num(s: &str) -> Option<u32> {
    let l = s.trim_end_matches('.').to_ascii_lowercase();
    Some(match l.get(..3)? {
        "jan" => 1,
        "feb" => 2,
        "mar" => 3,
        "apr" => 4,
        "may" => 5,
        "jun" => 6,
        "jul" => 7,
        "aug" => 8,
        "sep" => 9,
        "oct" => 10,
        "nov" => 11,
        "dec" => 12,
        _ => return None,
    })
}

fn weekday_of(s: &str) -> Option<Weekday> {
    Some(match s.to_ascii_lowercase().get(..3)? {
        "mon" => Weekday::Mon,
        "tue" => Weekday::Tue,
        "wed" => Weekday::Wed,
        "thu" => Weekday::Thu,
        "fri" => Weekday::Fri,
        "sat" => Weekday::Sat,
        "sun" => Weekday::Sun,
        _ => return None,
    })
}

fn num_of(s: &str) -> Option<i64> {
    let l = s.to_ascii_lowercase();
    if l.bytes().all(|b| b.is_ascii_digit()) {
        return l.parse().ok();
    }
    if l.contains("couple") {
        return Some(2);
    }
    Some(match l.as_str() {
        "a" | "an" | "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        "twenty" => 20,
        _ => return None,
    })
}

fn unit_of(s: &str) -> Granularity {
    match s.to_ascii_lowercase().as_str() {
        "day" => Granularity::Day,
        "week" => Granularity::Week,
        "month" => Granularity::Month,
        _ => Granularity::Year,
    }
}

/// The calendar unit containing `d` (the day, its week, month or year).
fn unit_iv(d: NaiveDate, g: Granularity) -> Option<(NaiveDate, NaiveDate)> {
    match g {
        Granularity::Day => Some((d, days(d, 1))),
        Granularity::Week => {
            let w = week_start(d);
            Some((w, days(w, 7)))
        }
        Granularity::Month => month_iv(d.year(), d.month()),
        Granularity::Year => year_iv(d.year()),
        Granularity::Season => {
            let (s, y) = season_of(d);
            season_iv(s, y)
        }
    }
}

/// `d` moved by `n` units (calendar months and years clamp the day).
fn shift(d: NaiveDate, g: Granularity, n: i64) -> Option<NaiveDate> {
    let months = |k: i64| -> Option<NaiveDate> {
        if k >= 0 {
            d.checked_add_months(Months::new(k as u32))
        } else {
            d.checked_sub_months(Months::new((-k) as u32))
        }
    };
    match g {
        Granularity::Day => Some(days(d, n)),
        Granularity::Week => Some(days(d, 7 * n)),
        Granularity::Month => months(n),
        Granularity::Season => months(3 * n),
        Granularity::Year => months(12 * n),
    }
}

fn two_digit_year(y: i32) -> i32 {
    if y < 70 {
        2000 + y
    } else {
        1900 + y
    }
}

// ------------------------------------------------------------------ patterns

const FULL: &str = "January|February|March|April|May|June|July|August|September|October|November|December";
const ABBR: &str = "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec";
const DAYN: &str = r"3[01]|[12][0-9]|0?[1-9]";
const ORD: &str = r"(?:st|nd|rd|th)?";
const Y4: &str = r"19[0-9][0-9]|20[0-9][0-9]|2100";
const NUM: &str = r"[0-9]{1,3}|a\s+couple\s+of|a\s+couple|couple\s+of|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty";
const WDS: &str = "monday|tuesday|wednesday|thursday|friday|saturday|sunday";
/// Capitalised weekday abbreviations, only after last / next / this.
const WD_ABBR: &str = "Mon|Tues|Tue|Wed|Thurs|Thur|Thu|Fri|Sat|Sun";
const UNIT: &str = "day|week|month|year";
const SEASONS: &str = "summer|winter|spring|fall|autumn";
const PART: &str = r"(?:\s+(?i:morning|afternoon|evening|night))?";

fn mon(g: &str) -> String {
    format!(r"\b(?P<{g}>{FULL}|{ABBR})(?:\.|\b)")
}

fn re(s: &str) -> Regex {
    Regex::new(s).unwrap_or_else(|e| panic!("memory_time regex {s}: {e}"))
}

static FUTURE: LazyLock<Regex> = LazyLock::new(|| {
    re(r"(?i)\b(?:will|won't|wont|going|gonna|plan|plans|planning|next|tomorrow|upcoming|hope\s+to|hoping\s+to|wanna|shall|intend|intends|scheduled|soon|can't\s+wait|cannot\s+wait|looking\s+forward|want\s+to|would\s+like|how\s+about|what\s+about|are\s+you\s+free|let['’]?s|works?\s+for|sounds?\s+good)\b|['’]ll\b")
});
/// Past markers that keep a question about a weekday in the past ("did you go on Friday?").
static PAST_Q: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)\b(?:did|was|were|had|went|been|got|saw)\b|\w+ed\b"));
static PERFECT: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)\b(?:have|has)\b|['’]ve\b|['’]s\s+been\b"));
static THE_BEFORE: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)\bthe\s+$"));
static OF_AFTER: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)^\s+of\b"));
static RANGE_WORD_BEFORE: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)\b(?:or|to)\s*$|[-–]\s*$"));
static NOT_A_DAY_BEFORE: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)\b(?:good|black|cyber|girl|man)\s+$"));
/// A capitalised word right before a bare weekday that is not a cue or a
/// sentence start ("Mental Health Monday").
static NAME_BEFORE: LazyLock<Regex> = LazyLock::new(|| {
    re(r#"[a-z,;:]\s+["“']?(?:[A-Z][a-z]+\s+)*[A-Z][a-z]+\s+$"#)
});
static HABIT_AFTER: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)^\s+(?:morning|afternoon|evening|night)s\b"));
static RATE_BEFORE: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)\b(?:once|twice|times)\s+$"));
static WRAP: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)\b(?P<w>ever\s+since|since|up\s+until|until|till|til|through|thru)\s+$"));
static CONNECT: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)^\s*(?P<c>-|–|—|to|through|thru|until|till|and)\s*$"));
static FROM_BEFORE: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)\b(?P<p>from|between)\s+$"));

type Handler = fn(&Cx<'_>, &Captures<'_>) -> Option<Cand>;

static PATTERNS: LazyLock<Vec<(Regex, Handler)>> = LazyLock::new(|| {
    let v: Vec<(String, Handler)> = vec![
        // ranges inside one date: May 3-7, 3–7 May 2023, from March to May
        (
            format!(r"{}\s*(?P<d1>{DAYN}){ORD}\s*(?:-|–|—|(?i:to|through|until|till))\s*(?P<d2>{DAYN}){ORD}\b(?:(?:,\s*|\s+)(?P<y>{Y4})\b)?", mon("m")),
            h_day_range,
        ),
        (
            format!(r"\b(?P<d1>{DAYN}){ORD}\s*(?:-|–|—|(?i:to))\s*(?P<d2>{DAYN}){ORD}(?:\s+of)?\s+{}(?:(?:,\s*|\s+)(?P<y>{Y4})\b)?", mon("m")),
            h_day_range,
        ),
        (
            format!(r"\b(?P<pre>(?i:from|between))\s+(?P<m1>{FULL})(?:,?\s+(?P<y1>{Y4}))?\s*(?:-|–|—|(?i:to|and|through|until|till))\s*(?P<m2>{FULL})\b(?:,?\s+(?P<y2>{Y4})\b)?"),
            h_month_range,
        ),
        (
            format!(r"(?:\b(?P<pre>(?i:from|between))\s+)?\b(?P<y1>{Y4})\s*(?P<c>-|–|—|(?i:to|and|through|until|till))\s*(?P<y2>{Y4}|[0-9][0-9])\b"),
            h_year_range,
        ),
        // absolute dates
        (
            format!(r"(?:\b(?P<wd>(?i:{WDS})),?\s+)?{}\s*(?P<d>{DAYN}){ORD}\b(?:(?:,\s*|\s+)(?:of\s+)?(?P<y>{Y4})\b)?", mon("m")),
            h_month_day,
        ),
        (
            format!(r"\b(?:(?i:the)\s+)?(?P<d>{DAYN}){ORD}(?:\s+of)?\s+{}(?:(?:,\s*|\s+)(?P<y>{Y4})\b)?", mon("m")),
            h_month_day,
        ),
        (
            r"\b(?P<y>[0-9]{4})(?P<s1>[-/.])(?P<m>[0-9]{1,2})(?P<s2>[-/.])(?P<d>[0-9]{1,2})\b".to_string(),
            h_iso,
        ),
        (r"\b(?P<a>[0-9]{1,2})/(?P<b>[0-9]{1,2})/(?P<y>[0-9]{4}|[0-9]{2})\b".to_string(), h_us),
        (format!(r"{}(?:,?\s+|\s+of\s+)(?P<y>{Y4})\b", mon("m")), h_month_year),
        (
            format!(r"\b(?P<cue>(?i:the\s+end\s+of|end\s+of|the\s+beginning\s+of|beginning\s+of|the\s+start\s+of|start\s+of|the\s+middle\s+of|middle\s+of|back\s+in|in|since|until|till|through|during|throughout|last|this|next|early|late|mid))[\s-]+(?P<m>{FULL})\b"),
            h_month_cue,
        ),
        (
            format!(r"\b(?P<cue>(?i:back\s+in|in|since|until|till|through|during|throughout|from|early|late|mid|the\s+year))[\s-]+(?P<y>{Y4})\b"),
            h_year_cue,
        ),
        (format!(r"\b(?P<pre>(?i:last|this|next))\s+Christmas\b"), h_christmas),
        (format!(r"\bChristmas(?P<eve>\s+(?:Day|Eve))?(?:,?\s+|\s+of\s+)(?P<y>{Y4})\b"), h_christmas),
        // relative
        (
            r"(?i)\b(?P<a>the\s+day\s+before\s+yesterday|the\s+day\s+after\s+tomorrow|yesterday|today|tonight|tomorrow|last\s+night|this\s+morning|this\s+afternoon|this\s+evening)(?:\s+(?:morning|afternoon|evening|night))?\b".to_string(),
            h_rel_day,
        ),
        (format!(r"(?i)\b(?P<n>{NUM})\s+(?P<u>{UNIT})s?\s+ago\b"), h_ago),
        (format!(r"(?i)\bin\s+(?P<n>{NUM})\s+(?P<u>{UNIT})s?\b(?:['’]s?\s+time\b)?"), h_in),
        (format!(r"(?i)\b(?P<n>{NUM})\s+(?P<u>{UNIT})s?\s+from\s+now\b"), h_in),
        (
            format!(r"(?:\b(?P<pre>(?i:last|next|this\s+past|this\s+coming|this|on|every|each|that|most|since|until|till|by))\s+)?\b(?P<wd>(?i:{WDS}))(?P<part>{PART})\b"),
            h_weekday,
        ),
        (
            format!(r"\b(?P<pre>(?i:last|next|this\s+past|this\s+coming|this))\s+(?P<wd>{WD_ABBR})\b(?P<part>{PART})"),
            h_weekday,
        ),
        (r"(?i)\b(?P<pre>this\s+past|this\s+coming|this|last|next|over\s+the)\s+weekend\b".to_string(), h_weekend),
        (format!(r"(?i)\b(?P<n>{NUM})\s+weekends?\s+ago\b"), h_weekends_ago),
        (
            r"(?i)\b(?P<pre>earlier\s+this|later\s+this|early\s+this|late\s+this|early\s+last|late\s+last|early\s+next|late\s+next|last|this|next)\s+(?P<u>week|month|year)\b".to_string(),
            h_rel_unit,
        ),
        (r"(?i)\bthe\s+(?P<u>week|month|year)\s+(?P<w>before\s+last|after\s+next)\b".to_string(), h_rel_unit2),
        (format!(r"(?i)\b(?P<pre>this\s+past|this\s+coming|last|this|next)\s+(?P<s>{SEASONS})\b"), h_season_rel),
        (format!(r"\b(?:(?i:the)\s+)?(?P<s>(?i:{SEASONS}))(?:\s+of)?\s+(?P<y>{Y4})\b"), h_season_year),
        // durations
        (
            format!(r"(?i)\b(?:in|over|during|for|within)\s+the\s+(?:past|last)\s+(?:(?P<n>{NUM})\s+)?(?P<u>{UNIT})s?\b"),
            h_past_n,
        ),
        (format!(r"(?i)\bfor\s+(?P<n>{NUM})\s+(?P<u>{UNIT})s?(?P<now>\s+now)?\b"), h_for_n),
    ];
    v.into_iter().map(|(s, h)| (re(&s), h)).collect()
});

// ------------------------------------------------------------------ matching

struct Cx<'a> {
    t: &'a str,
    a: NaiveDate,
}

#[derive(Debug, Clone)]
struct Cand {
    s: usize,
    e: usize,
    start: NaiveDate,
    end: NaiveDate,
    g: Granularity,
    k: TimeKind,
    /// an absolute date or year (may join a dash range without "from")
    abs: bool,
}

impl Cx<'_> {
    /// Up to `n` bytes before `s`, cut at a char boundary.
    fn before(&self, s: usize, n: usize) -> &str {
        let mut lo = s.saturating_sub(n);
        while !self.t.is_char_boundary(lo) {
            lo -= 1;
        }
        &self.t[lo..s]
    }

    fn after(&self, e: usize, n: usize) -> &str {
        let mut hi = (e + n).min(self.t.len());
        while !self.t.is_char_boundary(hi) {
            hi += 1;
        }
        &self.t[e..hi]
    }

    fn sentence(&self, s: usize, e: usize) -> &str {
        let stop = |c: char| matches!(c, '.' | '!' | '?' | '\n');
        let b = self.t[..s].rfind(stop).map_or(0, |i| i + 1);
        let f = self.t[e..].find(stop).map_or(self.t.len(), |i| e + i);
        &self.t[b..f]
    }

    /// A future cue in the sentence, or a question with no past marker
    /// ("Does Saturday sound good?").
    fn future(&self, s: usize, e: usize) -> bool {
        let sent = self.sentence(s, e);
        if FUTURE.is_match(sent) {
            return true;
        }
        let end = sent.as_ptr() as usize - self.t.as_ptr() as usize + sent.len();
        self.t[end..].starts_with('?') && !PAST_Q.is_match(sent)
    }

    fn the_before(&self, s: usize) -> bool {
        THE_BEFORE.is_match(self.before(s, 8))
    }

    fn of_after(&self, e: usize) -> bool {
        OF_AFTER.is_match(self.after(e, 6))
    }

    /// No digit, slash, currency or numeric punctuation glued to a numeric
    /// match (versions, prices, phone numbers, scores, fractions).
    fn edges_ok(&self, s: usize, e: usize) -> bool {
        let mut prev = self.t[..s].chars().rev();
        if let Some(p) = prev.next() {
            if p.is_ascii_digit() || "/$£€#".contains(p) {
                return false;
            }
            if ".-:,+".contains(p) && prev.next().is_some_and(|q| q.is_ascii_digit()) {
                return false;
            }
        }
        let mut next = self.t[e..].chars();
        if let Some(n) = next.next() {
            if n.is_ascii_digit() || "/%".contains(n) {
                return false;
            }
            if ".-:,–".contains(n) && next.next().is_some_and(|q| q.is_ascii_digit()) {
                return false;
            }
        }
        true
    }

    fn month_year(&self, m: u32, how: MonthHow) -> i32 {
        let (ay, am) = (self.a.year(), self.a.month());
        match how {
            MonthHow::Last => {
                if m < am {
                    ay
                } else {
                    ay - 1
                }
            }
            MonthHow::Next => {
                if m > am {
                    ay
                } else {
                    ay + 1
                }
            }
            MonthHow::This => ay,
            MonthHow::Past => {
                if m <= am {
                    ay
                } else {
                    ay - 1
                }
            }
            MonthHow::Future => {
                if m >= am {
                    ay
                } else {
                    ay + 1
                }
            }
        }
    }

    /// A month and day without a year: the nearest occurrence (ties to the
    /// past), or the first on or after the anchor day with a future cue.
    /// (The latest one before the anchor in a past-tense sentence was
    /// tried: LongMemEval's session dates often precede the events their
    /// messages report, and it moved 59 of its dates a year back.)
    fn md_year(&self, m: u32, d: u32, future: bool) -> Option<NaiveDate> {
        let ay = self.a.year();
        let opts: Vec<NaiveDate> = (ay - 1..=ay + 1).filter_map(|y| NaiveDate::from_ymd_opt(y, m, d)).collect();
        if future {
            return opts.into_iter().find(|x| *x >= self.a);
        }
        opts.into_iter()
            .min_by_key(|x| ((*x - self.a).num_days().abs(), (*x > self.a) as u8))
    }

    fn weekday(&self, w: Weekday, how: &str) -> NaiveDate {
        let a = self.a.weekday().num_days_from_monday() as i64;
        let t = w.num_days_from_monday() as i64;
        match how {
            "before" => {
                let d = (a - t).rem_euclid(7);
                days(self.a, -(if d == 0 { 7 } else { d }))
            }
            "after" => {
                let d = (t - a).rem_euclid(7);
                days(self.a, if d == 0 { 7 } else { d })
            }
            _ => days(week_start(self.a), t),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MonthHow {
    Last,
    Next,
    This,
    Past,
    Future,
}

fn cand(s: usize, e: usize, iv: (NaiveDate, NaiveDate), g: Granularity, abs: bool) -> Cand {
    Cand {
        s,
        e,
        start: iv.0,
        end: iv.1,
        g,
        k: TimeKind::Point,
        abs,
    }
}

fn whole(c: &Captures<'_>) -> (usize, usize) {
    let m = c.get(0).unwrap();
    (m.start(), m.end())
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").to_ascii_lowercase()
}

fn year_of(c: &Captures<'_>, g: &str) -> Option<i32> {
    c.name(g).and_then(|y| y.as_str().parse().ok())
}

fn h_month_day(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    let mname = c.name("m")?.as_str();
    let m = month_num(mname)?;
    let d: u32 = c.name("d")?.as_str().parse().ok()?;
    let y = year_of(c, "y");
    if y.is_none() && mname == "Jan" {
        return None; // "Jan" is also a name
    }
    if !cx.edges_ok(s, e) {
        return None;
    }
    let date = match y {
        Some(y) => NaiveDate::from_ymd_opt(y, m, d)?,
        None => cx.md_year(m, d, cx.future(s, e))?,
    };
    Some(cand(s, e, (date, days(date, 1)), Granularity::Day, true))
}

fn h_day_range(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    let mname = c.name("m")?.as_str();
    let m = month_num(mname)?;
    let d1: u32 = c.name("d1")?.as_str().parse().ok()?;
    let d2: u32 = c.name("d2")?.as_str().parse().ok()?;
    if d2 <= d1 || !cx.edges_ok(s, e) || (mname == "Jan" && c.name("y").is_none()) {
        return None;
    }
    let a = match year_of(c, "y") {
        Some(y) => NaiveDate::from_ymd_opt(y, m, d1)?,
        None => cx.md_year(m, d1, cx.future(s, e))?,
    };
    let b = NaiveDate::from_ymd_opt(a.year(), m, d2)?;
    let mut x = cand(s, e, (a, days(b, 1)), Granularity::Day, true);
    x.k = TimeKind::Range;
    Some(x)
}

fn h_month_range(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    let m1 = month_num(c.name("m1")?.as_str())?;
    let m2 = month_num(c.name("m2")?.as_str())?;
    let (y1, y2) = match (year_of(c, "y1"), year_of(c, "y2")) {
        (Some(a), Some(b)) => (a, b),
        (Some(a), None) => (a, if m2 >= m1 { a } else { a + 1 }),
        (None, Some(b)) => (if m1 <= m2 { b } else { b - 1 }, b),
        (None, None) => {
            let how = if cx.future(s, e) { MonthHow::Future } else { MonthHow::Past };
            let a = cx.month_year(m1, how);
            (a, if m2 >= m1 { a } else { a + 1 })
        }
    };
    let a = month_iv(y1, m1)?;
    let b = month_iv(y2, m2)?;
    if b.1 <= a.0 {
        return None;
    }
    let mut x = cand(s, e, (a.0, b.1), Granularity::Month, true);
    x.k = TimeKind::Range;
    Some(x)
}

fn h_year_range(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    let pre = c.name("pre").map(|p| norm(p.as_str()));
    let conn = norm(c.name("c")?.as_str());
    let dash = matches!(conn.as_str(), "-" | "–" | "—");
    if conn == "and" && pre.as_deref() != Some("between") {
        return None;
    }
    if pre.is_none() && !dash {
        return None;
    }
    let y1: i32 = c.name("y1")?.as_str().parse().ok()?;
    let y2s = c.name("y2")?.as_str();
    let y2: i32 = if y2s.len() == 2 {
        y1 / 100 * 100 + y2s.parse::<i32>().ok()?
    } else {
        y2s.parse().ok()?
    };
    if y2 <= y1 || y2 - y1 > 100 || !cx.edges_ok(s, e) {
        return None;
    }
    let mut x = cand(s, e, (year_iv(y1)?.0, year_iv(y2)?.1), Granularity::Year, true);
    x.k = TimeKind::Range;
    Some(x)
}

fn h_iso(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    if c.name("s1")?.as_str() != c.name("s2")?.as_str() || !cx.edges_ok(s, e) {
        return None;
    }
    let y: i32 = c.name("y")?.as_str().parse().ok()?;
    if !(1900..=2100).contains(&y) {
        return None;
    }
    let d = NaiveDate::from_ymd_opt(y, c.name("m")?.as_str().parse().ok()?, c.name("d")?.as_str().parse().ok()?)?;
    Some(cand(s, e, (d, days(d, 1)), Granularity::Day, true))
}

fn h_us(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    if !cx.edges_ok(s, e) {
        return None;
    }
    let a: u32 = c.name("a")?.as_str().parse().ok()?;
    let b: u32 = c.name("b")?.as_str().parse().ok()?;
    let ys = c.name("y")?.as_str();
    let y: i32 = ys.parse().ok()?;
    let y = if ys.len() == 2 { two_digit_year(y) } else { y };
    if !(1900..=2100).contains(&y) {
        return None;
    }
    let (m, d) = if a > 12 { (b, a) } else { (a, b) };
    let date = NaiveDate::from_ymd_opt(y, m, d)?;
    Some(cand(s, e, (date, days(date, 1)), Granularity::Day, true))
}

fn h_month_year(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    if !cx.edges_ok(s, e) {
        return None;
    }
    let m = month_num(c.name("m")?.as_str())?;
    let y: i32 = c.name("y")?.as_str().parse().ok()?;
    Some(cand(s, e, month_iv(y, m)?, Granularity::Month, true))
}

fn h_month_cue(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s0, e) = whole(c);
    let cue = norm(c.name("cue")?.as_str());
    let mg = c.name("m")?;
    let m = month_num(mg.as_str())?;
    let (s, how) = match cue.as_str() {
        "in" | "during" | "throughout" | "back in" => {
            (mg.start(), if cx.future(s0, e) { MonthHow::Future } else { MonthHow::Past })
        }
        "since" => (mg.start(), MonthHow::Past),
        "until" | "till" | "through" => (mg.start(), MonthHow::Future),
        "last" => (s0, MonthHow::Last),
        "next" => (s0, MonthHow::Next),
        "this" => (s0, MonthHow::This),
        _ => (s0, if cx.future(s0, e) { MonthHow::Future } else { MonthHow::Past }),
    };
    let y = cx.month_year(m, how);
    Some(cand(s, e, month_iv(y, m)?, Granularity::Month, false))
}

fn h_year_cue(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s0, e) = whole(c);
    let cue = norm(c.name("cue")?.as_str());
    let yg = c.name("y")?;
    if !cx.edges_ok(yg.start(), e) {
        return None;
    }
    let s = match cue.as_str() {
        "early" | "late" | "mid" | "the year" => s0,
        _ => yg.start(),
    };
    let y: i32 = yg.as_str().parse().ok()?;
    Some(cand(s, e, year_iv(y)?, Granularity::Year, true))
}

fn h_christmas(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    let day = if c.name("eve").is_some_and(|x| x.as_str().trim() == "Eve") { 24 } else { 25 };
    let d = match (year_of(c, "y"), c.name("pre").map(|p| norm(p.as_str()))) {
        (Some(y), _) => NaiveDate::from_ymd_opt(y, 12, day)?,
        (None, Some(p)) => {
            let ay = cx.a.year();
            let this = NaiveDate::from_ymd_opt(ay, 12, day)?;
            match p.as_str() {
                "last" => {
                    if this < cx.a {
                        this
                    } else {
                        NaiveDate::from_ymd_opt(ay - 1, 12, day)?
                    }
                }
                "next" => {
                    if this > cx.a {
                        this
                    } else {
                        NaiveDate::from_ymd_opt(ay + 1, 12, day)?
                    }
                }
                _ => this,
            }
        }
        _ => return None,
    };
    Some(cand(s, e, (d, days(d, 1)), Granularity::Day, true))
}

fn h_rel_day(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    let a = norm(c.name("a")?.as_str());
    let off = match a.as_str() {
        "the day before yesterday" => -2,
        "the day after tomorrow" => 2,
        "yesterday" => -1,
        "tomorrow" => 1,
        "last night" => {
            if cx.the_before(s) {
                return None; // "the last night of the trip"
            }
            -1
        }
        _ => 0,
    };
    let d = days(cx.a, off);
    Some(cand(s, e, (d, days(d, 1)), Granularity::Day, false))
}

fn h_ago(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    if RANGE_WORD_BEFORE.is_match(cx.before(s, 6)) || !cx.edges_ok(s, e) {
        return None; // "two or three weeks ago"
    }
    let n = num_of(c.name("n")?.as_str())?;
    let g = unit_of(c.name("u")?.as_str());
    let d = shift(cx.a, g, -n)?;
    Some(cand(s, e, unit_iv(d, g)?, g, false))
}

fn h_in(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    if RANGE_WORD_BEFORE.is_match(cx.before(s, 6)) || RATE_BEFORE.is_match(cx.before(s, 8)) {
        return None;
    }
    let n = num_of(c.name("n")?.as_str())?;
    let g = unit_of(c.name("u")?.as_str());
    let in_form = c.get(0)?.as_str().to_ascii_lowercase().starts_with("in");
    if in_form && n == 1 && matches!(g, Granularity::Day | Granularity::Year) {
        return None; // "twice in a day", "once in a year"
    }
    let d = shift(cx.a, g, n)?;
    Some(cand(s, e, unit_iv(d, g)?, g, false))
}

fn h_weekday(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s0, e) = whole(c);
    let wg = c.name("wd")?;
    let w = weekday_of(wg.as_str())?;
    let pre = c.name("pre").map(|p| norm(p.as_str()));
    if HABIT_AFTER.is_match(cx.after(e, 12)) {
        return None; // "Saturday mornings": a habit
    }
    let (s, how) = match pre.as_deref() {
        Some("every" | "each" | "that" | "most") => return None,
        Some("last" | "this past") => (s0, "before"),
        Some("next" | "this coming") => (s0, "after"),
        // "this Saturday" said on a Sunday about a plan: the coming one
        Some("this") => (
            s0,
            if cx.weekday(w, "this") < cx.a && !PAST_Q.is_match(cx.sentence(s0, e)) { "after" } else { "this" },
        ),
        Some("since") => (wg.start(), "before"),
        Some("until" | "till" | "by") => (wg.start(), "after"),
        _ => {
            if NOT_A_DAY_BEFORE.is_match(cx.before(wg.start(), 8)) {
                return None; // Good Friday, Black Friday
            }
            if pre.is_none() && NAME_BEFORE.is_match(cx.before(wg.start(), 24)) {
                return None; // "Mental Health Monday", "Taco Tuesday": a name
            }
            let s = if pre.is_some() { wg.start() } else { s0 };
            (s, if cx.future(s0, e) { "after" } else { "before" })
        }
    };
    let d = cx.weekday(w, how);
    Some(cand(s, e, (d, days(d, 1)), Granularity::Day, false))
}

fn h_weekend(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    let pre = norm(c.name("pre")?.as_str());
    if cx.of_after(e) || (matches!(pre.as_str(), "last" | "next") && cx.the_before(s)) {
        return None;
    }
    let this = days(week_start(cx.a), 5);
    let past = if this >= cx.a { days(this, -7) } else { this };
    let sat = match pre.as_str() {
        "this" => this,
        "last" => {
            if days(this, 1) >= cx.a {
                days(this, -7)
            } else {
                this
            }
        }
        "next" => days(this, 7),
        "this past" => past,
        "this coming" => {
            if this <= cx.a {
                days(this, 7)
            } else {
                this
            }
        }
        _ => {
            if cx.future(s, e) {
                if this < cx.a {
                    days(this, 7)
                } else {
                    this
                }
            } else {
                past
            }
        }
    };
    Some(cand(s, e, (sat, days(sat, 2)), Granularity::Day, false))
}

fn h_weekends_ago(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    if RANGE_WORD_BEFORE.is_match(cx.before(s, 6)) {
        return None;
    }
    let n = num_of(c.name("n")?.as_str())?;
    let this = days(week_start(cx.a), 5);
    let past = if this >= cx.a { days(this, -7) } else { this };
    let sat = days(past, -7 * (n - 1));
    Some(cand(s, e, (sat, days(sat, 2)), Granularity::Day, false))
}

fn h_rel_unit(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    let pre = norm(c.name("pre")?.as_str());
    let g = unit_of(c.name("u")?.as_str());
    let n = if pre.ends_with("last") {
        -1
    } else if pre.ends_with("next") {
        1
    } else {
        0
    };
    if n != 0 && (cx.the_before(s) || cx.of_after(e)) {
        return None; // "the last week of school", "the next year"
    }
    let d = shift(cx.a, g, n)?;
    Some(cand(s, e, unit_iv(d, g)?, g, false))
}

fn h_rel_unit2(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    let g = unit_of(c.name("u")?.as_str());
    let n = if norm(c.name("w")?.as_str()).starts_with("before") { -2 } else { 2 };
    let d = shift(cx.a, g, n)?;
    Some(cand(s, e, unit_iv(d, g)?, g, false))
}

fn h_season_rel(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    let pre = norm(c.name("pre")?.as_str());
    if cx.the_before(s) || cx.of_after(e) {
        return None;
    }
    let x = season_name(c.name("s")?.as_str());
    let (cs, cy) = season_of(cx.a);
    let ay = cx.a.year();
    let iv = match pre.as_str() {
        "last" | "this past" => {
            if cs == x {
                season_iv(x, cy - 1)?
            } else {
                (ay - 2..=ay).rev().filter_map(|y| season_iv(x, y)).find(|iv| iv.1 <= cx.a)?
            }
        }
        "next" | "this coming" => {
            if cs == x {
                season_iv(x, cy + 1)?
            } else {
                (ay - 1..=ay + 1).filter_map(|y| season_iv(x, y)).find(|iv| iv.0 > cx.a)?
            }
        }
        _ => {
            if cs == x {
                season_iv(x, cy)?
            } else {
                season_iv(x, ay)?
            }
        }
    };
    Some(cand(s, e, iv, Granularity::Season, false))
}

fn h_season_year(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    if !cx.edges_ok(s, e) {
        return None;
    }
    let x = season_name(c.name("s")?.as_str());
    let y: i32 = c.name("y")?.as_str().parse().ok()?;
    Some(cand(s, e, season_iv(x, y)?, Granularity::Season, true))
}

fn duration(cx: &Cx<'_>, s: usize, e: usize, n: i64, g: Granularity) -> Option<Cand> {
    let start = shift(cx.a, g, -n)?;
    let mut x = cand(s, e, (start, days(cx.a, 1)), g, false);
    x.k = TimeKind::Duration;
    Some(x)
}

fn h_past_n(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    if cx.of_after(e) {
        return None; // "during the last two years of college"
    }
    let n = match c.name("n") {
        Some(n) => num_of(n.as_str())?,
        None => 1,
    };
    duration(cx, s, e, n, unit_of(c.name("u")?.as_str()))
}

fn h_for_n(cx: &Cx<'_>, c: &Captures<'_>) -> Option<Cand> {
    let (s, e) = whole(c);
    let now = c.name("now").is_some();
    let sent = cx.sentence(s, e);
    let off = sent.as_ptr() as usize - cx.t.as_ptr() as usize;
    let head = &sent[..s.saturating_sub(off).min(sent.len())];
    if !now && !PERFECT.is_match(head) {
        return None; // "for two weeks" is a length, not a time
    }
    let n = num_of(c.name("n")?.as_str())?;
    duration(cx, s, e, n, unit_of(c.name("u")?.as_str()))
}

/// `from X to Y`, `between X and Y`, `X – Y` (two absolute dates).
fn combine_ranges(cx: &Cx<'_>, cands: &mut Vec<Cand>) {
    let mut sorted: Vec<&Cand> = cands.iter().filter(|c| c.k == TimeKind::Point).collect();
    sorted.sort_by_key(|c| (c.s, c.e));
    let mut out = vec![];
    for a in &sorted {
        for b in &sorted {
            if b.s < a.e || b.s - a.e > 16 || !cx.t.is_char_boundary(a.e) {
                continue;
            }
            let Some(conn) = CONNECT.captures(&cx.t[a.e..b.s]) else { continue };
            let conn = norm(&conn["c"]);
            let pre = FROM_BEFORE.captures(cx.before(a.s, 10));
            let pre_w = pre.as_ref().map(|p| norm(&p["p"]));
            let ok = match (conn.as_str(), pre_w.as_deref()) {
                ("and", Some("between")) => true,
                ("and", _) => false,
                (_, Some("from")) => true,
                (_, Some("between")) => false,
                ("-" | "–" | "—" | "to" | "through" | "thru", None) => a.abs && b.abs,
                _ => false,
            };
            if !ok || b.end <= a.start {
                continue;
            }
            let s = match &pre {
                Some(p) => a.s - (cx.before(a.s, 10).len() - p.get(0).unwrap().start()),
                None => a.s,
            };
            out.push(Cand {
                s,
                e: b.e,
                start: a.start,
                end: b.end.max(a.end),
                g: a.g.max(b.g),
                k: TimeKind::Range,
                abs: false,
            });
        }
    }
    cands.extend(out);
}

/// `since X` and `until X` around a point X.
fn wrap_since_until(cx: &Cx<'_>, cands: &mut Vec<Cand>) {
    let anchor_end = days(cx.a, 1);
    let mut out = vec![];
    for c in cands.iter().filter(|c| c.k == TimeKind::Point) {
        let head = cx.before(c.s, 16);
        let Some(w) = WRAP.captures(head) else { continue };
        let s = c.s - (head.len() - w.get(0).unwrap().start());
        let word = norm(&w["w"]);
        let mut x = c.clone();
        x.s = s;
        if word.ends_with("since") {
            if c.start >= anchor_end {
                continue;
            }
            x.k = TimeKind::Since;
            x.end = anchor_end;
        } else {
            if c.end <= cx.a {
                continue; // a past "until": keep X itself
            }
            x.k = TimeKind::Until;
            x.start = cx.a;
        }
        out.push(x);
    }
    cands.extend(out);
}

fn maybe_temporal(t: &str) -> bool {
    if t.bytes().any(|b| b.is_ascii_digit()) {
        return true;
    }
    let l = t.to_ascii_lowercase();
    [
        "day", "night", "tomorrow", "week", "month", "year", "ago", "morning", "afternoon", "evening", "summer",
        "winter", "spring", "fall", "autumn", "christmas", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    ]
    .iter()
    .any(|w| l.contains(w))
}

/// All temporal expressions in `text`, resolved against `anchor_us` (when it
/// was said, µs), in text order, non-overlapping (the longest match wins).
pub fn parse_times(text: &str, anchor_us: i64) -> Vec<TimeExpr> {
    if !maybe_temporal(text) {
        return vec![];
    }
    let cx = Cx {
        t: text,
        a: date_of(anchor_us),
    };
    let mut cands: Vec<Cand> = vec![];
    for (r, h) in PATTERNS.iter() {
        for c in r.captures_iter(text) {
            if let Some(x) = h(&cx, &c) {
                if x.e > x.s && x.end > x.start {
                    cands.push(x);
                }
            }
        }
    }
    combine_ranges(&cx, &mut cands);
    wrap_since_until(&cx, &mut cands);
    // longest first, then leftmost; keep the non-overlapping ones
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(cands[i].e - cands[i].s), cands[i].s, i));
    let mut taken: Vec<&Cand> = vec![];
    for i in order {
        let c = &cands[i];
        if taken.iter().all(|t| c.e <= t.s || c.s >= t.e) {
            taken.push(c);
        }
    }
    taken.sort_by_key(|c| c.s);
    taken
        .into_iter()
        .map(|c| TimeExpr {
            start: us_of(c.start),
            end: us_of(c.end),
            granularity: c.g,
            kind: c.k,
            expr: text[c.s..c.e].to_string(),
            span: (c.s, c.e),
        })
        .collect()
}

// ------------------------------------------------------------------ rendering

fn ymd(d: NaiveDate) -> String {
    d.format("%Y-%m-%d").to_string()
}

fn label(d: NaiveDate, g: Granularity) -> String {
    match g {
        Granularity::Day => ymd(d),
        Granularity::Week => format!("week of {}", ymd(week_start(d))),
        Granularity::Month => d.format("%B %Y").to_string(),
        Granularity::Season => {
            let (s, y) = season_of(d);
            match s {
                Season::Spring => format!("spring {y}"),
                Season::Summer => format!("summer {y}"),
                Season::Fall => format!("fall {y}"),
                Season::Winter => format!("winter {}-{:02}", y, (y + 1).rem_euclid(100)),
            }
        }
        Granularity::Year => d.format("%Y").to_string(),
    }
}

/// Reader-facing rendering of the resolved interval: "2023-05-07 (Sun)",
/// "2023-05-06 to 2023-05-07", "week of 2023-05-01", "May 2023", "summer
/// 2022", "2019", "since 2019", "until June 2023", "2019 to 2021", and for
/// durations the days "2023-04-24 to 2023-05-08".
pub fn describe(e: &TimeExpr) -> String {
    let s = date_of(e.start);
    let last = date_of(e.end - DAY_US).max(s);
    let g = e.granularity;
    let span = |a: String, b: String| if a == b { a } else { format!("{a} to {b}") };
    match e.kind {
        TimeKind::Point => {
            if g == Granularity::Day && last == s {
                s.format("%Y-%m-%d (%a)").to_string()
            } else {
                span(label(s, g), label(last, g))
            }
        }
        TimeKind::Since => format!("since {}", label(s, g)),
        TimeKind::Until => format!("until {}", label(last, g)),
        TimeKind::Range => span(label(s, g), label(last, g)),
        TimeKind::Duration => span(ymd(s), ymd(last)),
    }
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    fn at(y: i32, m: u32, d: u32) -> i64 {
        Utc.with_ymd_and_hms(y, m, d, 13, 56, 0).unwrap().timestamp_micros()
    }

    /// Monday 2023-05-08, 13:56 UTC
    fn a() -> i64 {
        at(2023, 5, 8)
    }

    fn ds(us: i64) -> String {
        ymd(date_of(us))
    }

    fn got(text: &str, anchor: i64) -> Vec<(String, String, String, &'static str, &'static str)> {
        parse_times(text, anchor)
            .into_iter()
            .map(|e| (e.expr.clone(), ds(e.start), ds(e.end), e.granularity.as_str(), e.kind.as_str()))
            .collect()
    }

    #[track_caller]
    fn one_at(text: &str, anchor: i64, expr: &str, start: &str, end: &str, g: &str) {
        let v = got(text, anchor);
        assert_eq!(v.len(), 1, "{text:?}: {v:?}");
        assert_eq!((v[0].0.as_str(), v[0].1.as_str(), v[0].2.as_str(), v[0].3), (expr, start, end, g), "{text:?}");
    }

    #[track_caller]
    fn one(text: &str, expr: &str, start: &str, end: &str, g: &str) {
        one_at(text, a(), expr, start, end, g)
    }

    #[track_caller]
    fn kind(text: &str, k: &str) {
        let v = got(text, a());
        assert_eq!(v.len(), 1, "{text:?}: {v:?}");
        assert_eq!(v[0].4, k, "{text:?}");
    }

    #[track_caller]
    fn none(text: &str) {
        let v = got(text, a());
        assert!(v.is_empty(), "{text:?} should have no dates: {v:?}");
    }

    #[test]
    fn relative_days() {
        one("I went to a support group yesterday.", "yesterday", "2023-05-07", "2023-05-08", "day");
        one("Today was great", "Today", "2023-05-08", "2023-05-09", "day");
        one("see you tomorrow evening!", "tomorrow evening", "2023-05-09", "2023-05-10", "day");
        one("the day before yesterday it rained", "the day before yesterday", "2023-05-06", "2023-05-07", "day");
        one("the day after tomorrow", "the day after tomorrow", "2023-05-10", "2023-05-11", "day");
        one("I slept badly last night", "last night", "2023-05-07", "2023-05-08", "day");
        one("I ran this morning", "this morning", "2023-05-08", "2023-05-09", "day");
        one("We're going out tonight", "tonight", "2023-05-08", "2023-05-09", "day");
        one("yesterday afternoon we met", "yesterday afternoon", "2023-05-07", "2023-05-08", "day");
    }

    #[test]
    fn offsets() {
        one("two days ago", "two days ago", "2023-05-06", "2023-05-07", "day");
        one("I did it 10 days ago", "10 days ago", "2023-04-28", "2023-04-29", "day");
        one("a couple of days ago", "a couple of days ago", "2023-05-06", "2023-05-07", "day");
        one("a week ago", "a week ago", "2023-05-01", "2023-05-08", "week");
        one("3 weeks ago", "3 weeks ago", "2023-04-17", "2023-04-24", "week");
        one("a month ago", "a month ago", "2023-04-01", "2023-05-01", "month");
        one("two years ago", "two years ago", "2021-01-01", "2022-01-01", "year");
        one("in two weeks", "in two weeks", "2023-05-22", "2023-05-29", "week");
        one("in a week's time", "in a week's time", "2023-05-15", "2023-05-22", "week");
        one("3 days from now", "3 days from now", "2023-05-11", "2023-05-12", "day");
        one("in a month", "in a month", "2023-06-01", "2023-07-01", "month");
        // a leap day minus a year clamps to Feb 28
        one_at("a year ago", at(2024, 2, 29), "a year ago", "2023-01-01", "2024-01-01", "year");
        one_at("12 months ago", at(2024, 2, 29), "12 months ago", "2023-02-01", "2023-03-01", "month");
        one_at("a month ago", at(2023, 1, 15), "a month ago", "2022-12-01", "2023-01-01", "month");
    }

    #[test]
    fn weekdays() {
        one("last Friday", "last Friday", "2023-05-05", "2023-05-06", "day");
        one("next Friday", "next Friday", "2023-05-12", "2023-05-13", "day");
        one("this Friday", "this Friday", "2023-05-12", "2023-05-13", "day");
        one("this past Friday", "this past Friday", "2023-05-05", "2023-05-06", "day");
        one("I went hiking on Saturday.", "Saturday", "2023-05-06", "2023-05-07", "day");
        one("I'm going hiking on Saturday.", "Saturday", "2023-05-13", "2023-05-14", "day");
        one("We'll meet on Monday", "Monday", "2023-05-15", "2023-05-16", "day");
        one("I saw her on Monday", "Monday", "2023-05-01", "2023-05-02", "day");
        one("last Friday night was fun", "last Friday night", "2023-05-05", "2023-05-06", "day");
        one("on friday we had pizza", "friday", "2023-05-05", "2023-05-06", "day");
        one("my car broke down last Fri", "last Fri", "2023-05-05", "2023-05-06", "day");
        one("I joined a group last Tues.", "last Tues", "2023-05-02", "2023-05-03", "day");
        one("How about Wednesday at 4?", "Wednesday", "2023-05-10", "2023-05-11", "day");
        one("Let's meet Friday", "Friday", "2023-05-12", "2023-05-13", "day");
        one("Does Saturday sound good?", "Saturday", "2023-05-13", "2023-05-14", "day");
        one("Did you go on Saturday?", "Saturday", "2023-05-06", "2023-05-07", "day");
        one("Saturday works for me!", "Saturday", "2023-05-13", "2023-05-14", "day");
        // "this <weekday>" already past in the week, about a plan: the coming one
        one_at(
            "I'm thinking of doing a bike ride this Saturday.",
            at(2023, 11, 5),
            "this Saturday",
            "2023-11-11",
            "2023-11-12",
            "day",
        );
        one_at("I went there this Saturday.", at(2023, 11, 5), "this Saturday", "2023-11-04", "2023-11-05", "day");
        // a capitalised name before a bare weekday is not a date
        none("For my \"Mental Health Monday\" series, I share tips.");
        none("We do Taco Tuesday at work.");
        // a month and day without a year: the nearest one
        one_at(
            "I attended a charity golf tournament on July 17th.",
            at(2023, 2, 21),
            "July 17th",
            "2023-07-17",
            "2023-07-18",
            "day",
        );
        one("Hey Sam, Monday was fun.", "Monday", "2023-05-01", "2023-05-02", "day");
        none("I last sat down here");
        // a Sunday anchor
        one_at("last Sunday", at(2023, 5, 7), "last Sunday", "2023-04-30", "2023-05-01", "day");
        one_at("next Monday", at(2023, 5, 7), "next Monday", "2023-05-08", "2023-05-09", "day");
    }

    #[test]
    fn weekends() {
        one("this weekend", "this weekend", "2023-05-13", "2023-05-15", "day");
        one("last weekend", "last weekend", "2023-05-06", "2023-05-08", "day");
        one("next weekend", "next weekend", "2023-05-20", "2023-05-22", "day");
        one("I went camping over the weekend", "over the weekend", "2023-05-06", "2023-05-08", "day");
        one("this past weekend", "this past weekend", "2023-05-06", "2023-05-08", "day");
        one("two weekends ago", "two weekends ago", "2023-04-29", "2023-05-01", "day");
        // said on a Sunday: this weekend is the current one, last weekend the one before
        one_at("this weekend", at(2023, 5, 7), "this weekend", "2023-05-06", "2023-05-08", "day");
        one_at("last weekend", at(2023, 5, 7), "last weekend", "2023-04-29", "2023-05-01", "day");
        one_at("over the weekend", at(2023, 5, 7), "over the weekend", "2023-05-06", "2023-05-08", "day");
    }

    #[test]
    fn units() {
        one("last week", "last week", "2023-05-01", "2023-05-08", "week");
        one("this week", "this week", "2023-05-08", "2023-05-15", "week");
        one("earlier this week", "earlier this week", "2023-05-08", "2023-05-15", "week");
        one("next week", "next week", "2023-05-15", "2023-05-22", "week");
        one("last month", "last month", "2023-04-01", "2023-05-01", "month");
        one("next month", "next month", "2023-06-01", "2023-07-01", "month");
        one("last year", "last year", "2022-01-01", "2023-01-01", "year");
        one("this year's trip", "this year", "2023-01-01", "2024-01-01", "year");
        one("the year before last", "the year before last", "2021-01-01", "2022-01-01", "year");
        one("the week after next", "the week after next", "2023-05-22", "2023-05-29", "week");
        one_at("last month", at(2023, 1, 10), "last month", "2022-12-01", "2023-01-01", "month");
        one_at("next year", at(2023, 12, 31), "next year", "2024-01-01", "2025-01-01", "year");
    }

    #[test]
    fn seasons() {
        one("last summer", "last summer", "2022-06-01", "2022-09-01", "season");
        one("this summer", "this summer", "2023-06-01", "2023-09-01", "season");
        one("next summer", "next summer", "2023-06-01", "2023-09-01", "season");
        one("last winter", "last winter", "2022-12-01", "2023-03-01", "season");
        one_at("last summer", at(2023, 10, 15), "last summer", "2023-06-01", "2023-09-01", "season");
        one_at("last summer", at(2023, 7, 10), "last summer", "2022-06-01", "2022-09-01", "season");
        one_at("last winter", at(2023, 1, 15), "last winter", "2021-12-01", "2022-03-01", "season");
        one_at("this winter", at(2023, 1, 15), "this winter", "2022-12-01", "2023-03-01", "season");
        one_at("this winter", at(2022, 12, 20), "this winter", "2022-12-01", "2023-03-01", "season");
        one_at("next winter", at(2023, 2, 10), "next winter", "2023-12-01", "2024-03-01", "season");
        one("in the summer of 2022", "the summer of 2022", "2022-06-01", "2022-09-01", "season");
        one("fall 2021", "fall 2021", "2021-09-01", "2021-12-01", "season");
        one("Winter 2022 was cold", "Winter 2022", "2022-12-01", "2023-03-01", "season");
    }

    #[test]
    fn months() {
        one("in March", "March", "2023-03-01", "2023-04-01", "month");
        one("I'll visit in March", "March", "2024-03-01", "2024-04-01", "month");
        one("in May", "May", "2023-05-01", "2023-06-01", "month");
        one("last March", "last March", "2023-03-01", "2023-04-01", "month");
        one("last May", "last May", "2022-05-01", "2022-06-01", "month");
        one("next March", "next March", "2024-03-01", "2024-04-01", "month");
        one("this December", "this December", "2023-12-01", "2024-01-01", "month");
        one("early April", "early April", "2023-04-01", "2023-05-01", "month");
        one("mid-February", "mid-February", "2023-02-01", "2023-03-01", "month");
        one("at the end of January", "the end of January", "2023-01-01", "2023-02-01", "month");
        one("We're going to Spain in July", "July", "2023-07-01", "2023-08-01", "month");
        one("May 2023", "May 2023", "2023-05-01", "2023-06-01", "month");
        one("March of 2022", "March of 2022", "2022-03-01", "2022-04-01", "month");
        one("in March 2022", "March 2022", "2022-03-01", "2022-04-01", "month");
        one("Sept 2021", "Sept 2021", "2021-09-01", "2021-10-01", "month");
    }

    #[test]
    fn absolute_dates() {
        for (t, e) in [
            ("On May 7, 2023 we met", "May 7, 2023"),
            ("May 7th 2023", "May 7th 2023"),
            ("7 May 2023", "7 May 2023"),
            ("the 7th of May, 2023", "the 7th of May, 2023"),
            ("the 7th of May", "the 7th of May"),
            ("on May 7", "May 7"),
            ("2023-05-07", "2023-05-07"),
            ("2023/05/07", "2023/05/07"),
            ("5/7/2023", "5/7/2023"),
            ("05/07/23", "05/07/23"),
            ("Sunday, May 7", "Sunday, May 7"),
        ] {
            one(t, e, "2023-05-07", "2023-05-08", "day");
        }
        one("13/5/2023", "13/5/2023", "2023-05-13", "2023-05-14", "day");
        one("Feb. 3", "Feb. 3", "2023-02-03", "2023-02-04", "day");
        one("Sept 5, 2021", "Sept 5, 2021", "2021-09-05", "2021-09-06", "day");
        one("Jan 5, 2022", "Jan 5, 2022", "2022-01-05", "2022-01-06", "day");
        // nearest occurrence without a year; a future cue looks forward
        one("December 25", "December 25", "2022-12-25", "2022-12-26", "day");
        one("We will fly out on December 25", "December 25", "2023-12-25", "2023-12-26", "day");
        one_at("on December 30", at(2023, 1, 3), "December 30", "2022-12-30", "2022-12-31", "day");
        one("Christmas 2022", "Christmas 2022", "2022-12-25", "2022-12-26", "day");
        one("last Christmas", "last Christmas", "2022-12-25", "2022-12-26", "day");
        // invalid calendar dates are skipped
        none("February 30, 2023");
    }

    #[test]
    fn years() {
        one("in 2019", "2019", "2019-01-01", "2020-01-01", "year");
        one("back in 2019", "2019", "2019-01-01", "2020-01-01", "year");
        one("mid-2021", "mid-2021", "2021-01-01", "2022-01-01", "year");
        one("the year 2000", "the year 2000", "2000-01-01", "2001-01-01", "year");
    }

    #[test]
    fn since_until_durations() {
        one("since 2019", "since 2019", "2019-01-01", "2023-05-09", "year");
        kind("since 2019", "since");
        one("since March", "since March", "2023-03-01", "2023-05-09", "month");
        one("ever since last year", "ever since last year", "2022-01-01", "2023-05-09", "year");
        one("since yesterday", "since yesterday", "2023-05-07", "2023-05-09", "day");
        one("until June", "until June", "2023-05-08", "2023-07-01", "month");
        kind("until June", "until");
        one("till Friday", "till Friday", "2023-05-08", "2023-05-13", "day");
        // a past until keeps the point itself
        one("I lived there until 2019", "2019", "2019-01-01", "2020-01-01", "year");
        kind("I lived there until 2019", "point");
        one("for the past three weeks", "for the past three weeks", "2023-04-17", "2023-05-09", "week");
        kind("for the past three weeks", "duration");
        one("I've been learning piano for 3 years", "for 3 years", "2020-05-08", "2023-05-09", "year");
        one("for two years now", "for two years now", "2021-05-08", "2023-05-09", "year");
        one("in the last 2 months", "in the last 2 months", "2023-03-08", "2023-05-09", "month");
        one("She has worked there for six months", "for six months", "2022-11-08", "2023-05-09", "month");
        none("I went to Spain for two weeks.");
        none("We stayed for three days");
    }

    #[test]
    fn ranges() {
        one("between 2019 and 2021", "between 2019 and 2021", "2019-01-01", "2022-01-01", "year");
        kind("between 2019 and 2021", "range");
        one("2019-2021", "2019-2021", "2019-01-01", "2022-01-01", "year");
        one("from 2019 to 2021", "from 2019 to 2021", "2019-01-01", "2022-01-01", "year");
        one("from May 3 to May 7", "from May 3 to May 7", "2023-05-03", "2023-05-08", "day");
        one("May 3 to May 7", "May 3 to May 7", "2023-05-03", "2023-05-08", "day");
        one("May 3-7", "May 3-7", "2023-05-03", "2023-05-08", "day");
        one("May 3 – 7, 2023", "May 3 – 7, 2023", "2023-05-03", "2023-05-08", "day");
        one("3–7 May 2023", "3–7 May 2023", "2023-05-03", "2023-05-08", "day");
        one("from March to May", "from March to May", "2023-03-01", "2023-06-01", "month");
        one("between June and August 2022", "between June and August 2022", "2022-06-01", "2022-09-01", "month");
        // Thursday 2023-05-11
        one_at("from Monday to Wednesday", at(2023, 5, 11), "from Monday to Wednesday", "2023-05-08", "2023-05-11", "day");
        kind("from Monday to Wednesday", "range");
    }

    #[test]
    fn several_in_text_order() {
        let v = got("Yesterday I met Anna, and next week we go to Rome. We first met in 2019.", a());
        let exprs: Vec<&str> = v.iter().map(|x| x.0.as_str()).collect();
        assert_eq!(exprs, ["Yesterday", "next week", "2019"]);
        let t = "On May 7, 2023 and last Friday";
        let v = parse_times(t, a());
        assert_eq!(v.len(), 2);
        for e in &v {
            assert_eq!(&t[e.span.0..e.span.1], e.expr);
        }
        assert_eq!(v[0].expr, "May 7, 2023");
        assert_eq!(v[1].expr, "last Friday");
    }

    #[test]
    fn unicode_is_safe() {
        let t = "😀 Yesterday — I visited Zoë’s café… on May 7, 2023 ✨ and I’ll go next Friday 🎉";
        let v = parse_times(t, a());
        let exprs: Vec<&str> = v.iter().map(|x| x.expr.as_str()).collect();
        assert_eq!(exprs, ["Yesterday", "May 7, 2023", "next Friday"]);
        for e in &v {
            assert_eq!(&t[e.span.0..e.span.1], e.expr);
        }
        for t in ["é2019", "since—2019", "in 2019é", "since 😀2019", "从2019年", "–", "May—", "ago😀", "in ’19"] {
            let _ = parse_times(t, a());
        }
        // every prefix and suffix of a mixed text, cut at char boundaries
        let t = "Ünïcödé last Friday’s 🎉 since 2019 – May 3–7, 2023 €5";
        for (i, _) in t.char_indices() {
            let _ = parse_times(&t[i..], a());
            let _ = parse_times(&t[..i], a());
        }
    }

    #[test]
    fn negatives() {
        for t in [
            "You may want to bring a jacket.",
            "May I ask you something?",
            "May be later.",
            "I may go in may.",
            "They march on the capital.",
            "March on, soldiers!",
            "This is the second time I've tried.",
            "Wait a second.",
            "First, let me explain.",
            "August is my cousin's name.",
            "I met August at the party.",
            "Prices fall in winter.",
            "I love spring flowers.",
            "June is coming over later.",
            "April and I went shopping.",
            "Summer is my favourite season.",
            "I spent a summer in Spain.",
            "The fall was painful.",
            "I finished just in time.",
            "It took a long time.",
            "One day I'll be famous.",
            "Meet me at 5 pm.",
            "The night was cold.",
            "I grew up in the 90s.",
            "Version 2.0.2023 is out.",
            "It costs $2019.",
            "Call me at 555-2019-2021.",
            "The score was 3-2.",
            "We're open 24/7.",
            "Never forget 9/11.",
            "Add 1/2 cup of sugar.",
            "I scored 2019 points.",
            "I sat on the sun lounger on wed",
            "In a few days we will know.",
            "I saw it the other day.",
            "I've been there recently.",
            "That was a while ago.",
            "She plays every Friday.",
            "We meet on Fridays.",
            "I prefer Saturday mornings for running.",
            "We spent two weeks in Spain.",
            "The last week of school was fun.",
            "Out of 2000 entries, ten won.",
            "It increased by 2020 units.",
            "It happened two or three weeks ago.",
            "Room 1905 is great.",
            "It happens twice in a day.",
            "Jan 5 times said no.",
            "the next year will tell",
            "Good Friday service",
            "I tried a few weeks ago",
            "Read chapter 3 of 12.",
        ] {
            none(t);
        }
    }

    #[test]
    fn describe_outputs() {
        let d = |t: &str, anchor: i64| -> String { describe(&parse_times(t, anchor)[0]) };
        assert_eq!(d("yesterday", a()), "2023-05-07 (Sun)");
        assert_eq!(d("this weekend", a()), "2023-05-13 to 2023-05-14");
        assert_eq!(d("last week", a()), "week of 2023-05-01");
        assert_eq!(d("a month ago", a()), "April 2023");
        assert_eq!(d("last summer", a()), "summer 2022");
        assert_eq!(d("last winter", a()), "winter 2022-23");
        assert_eq!(d("in 2019", a()), "2019");
        assert_eq!(d("since 2019", a()), "since 2019");
        assert_eq!(d("until June", a()), "until June 2023");
        assert_eq!(d("between 2019 and 2021", a()), "2019 to 2021");
        assert_eq!(d("May 3-7", a()), "2023-05-03 to 2023-05-07");
        assert_eq!(d("for the past two weeks", a()), "2023-04-24 to 2023-05-08");
        assert_eq!(d("from March to May", a()), "March 2023 to May 2023");
    }

    #[test]
    fn enums_round_trip() {
        for g in [Granularity::Day, Granularity::Week, Granularity::Month, Granularity::Season, Granularity::Year] {
            assert_eq!(Granularity::parse(g.as_str()), Some(g));
        }
        for k in [TimeKind::Point, TimeKind::Since, TimeKind::Until, TimeKind::Range, TimeKind::Duration] {
            assert_eq!(TimeKind::parse(k.as_str()), Some(k));
        }
        assert!(Granularity::Day < Granularity::Year);
    }

    #[test]
    fn precheck_skips_plain_text() {
        assert!(!maybe_temporal("I like pizza and pasta."));
        assert!(parse_times("", a()).is_empty());
    }
}
