/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Schedules of durable workflows (docs/specs/durable-workflows.md,
//! "Schedules"): five-field cron in UTC (`min hour day-of-month month
//! day-of-week`, with `*`, lists, ranges and steps; day-of-week 0-7, Sunday is
//! 0 or 7) and the shorthands `@every <n>(s|m|h|d)`, `@hourly`, `@daily`,
//! `@weekly`, `@monthly`.

use chrono::{DateTime, Datelike, Duration, TimeZone, Timelike, Utc};
use miette::{bail, miette, Result};

/// A parsed schedule.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Cron {
    /// A fixed interval in µs.
    Every(i64),
    /// Allowed values per field (bit sets): minutes 0-59, hours 0-23,
    /// days 1-31, months 1-12, weekdays 0-6; `dom_any` / `dow_any` record a
    /// `*`, because cron ORs the two day fields when both are restricted.
    Fields {
        min: u64,
        hour: u64,
        dom: u64,
        month: u64,
        dow: u64,
        dom_any: bool,
        dow_any: bool,
    },
}

fn field(spec: &str, lo: u32, hi: u32, name: &str) -> Result<(u64, bool)> {
    let mut bits = 0u64;
    let any = spec == "*";
    for part in spec.split(',') {
        let (range, step) = match part.split_once('/') {
            Some((r, s)) => (
                r,
                s.parse::<u32>()
                    .ok()
                    .filter(|&x| x > 0)
                    .ok_or_else(|| miette!("cron {name}: bad step in '{part}'"))?,
            ),
            None => (part, 1),
        };
        let (a, b) = if range == "*" {
            (lo, hi)
        } else if let Some((a, b)) = range.split_once('-') {
            let a: u32 = a.parse().map_err(|_| miette!("cron {name}: bad value '{a}'"))?;
            let b: u32 = b.parse().map_err(|_| miette!("cron {name}: bad value '{b}'"))?;
            (a, b)
        } else {
            let a: u32 = range.parse().map_err(|_| miette!("cron {name}: bad value '{range}'"))?;
            // `5/15` means from 5 to the end, every 15
            (a, if step > 1 { hi } else { a })
        };
        if a < lo || b > hi || a > b {
            bail!("cron {name}: '{part}' is outside {lo}-{hi}");
        }
        let mut v = a;
        while v <= b {
            bits |= 1 << v;
            v += step;
        }
    }
    Ok((bits, any))
}

fn every(spec: &str) -> Result<i64> {
    let spec = spec.trim();
    let (n, unit) = spec.split_at(spec.find(|c: char| !c.is_ascii_digit()).unwrap_or(spec.len()));
    let n: i64 = n.parse().map_err(|_| miette!("@every needs a number and a unit (s, m, h, d), got '{spec}'"))?;
    let us = match unit {
        "s" => 1_000_000,
        "m" => 60_000_000,
        "h" => 3_600_000_000,
        "d" => 86_400_000_000,
        _ => bail!("@every unit must be s, m, h or d, got '{unit}'"),
    };
    if n <= 0 {
        bail!("@every needs a positive interval");
    }
    Ok(n * us)
}

impl Cron {
    pub(crate) fn parse(spec: &str) -> Result<Cron> {
        let spec = spec.trim();
        let expanded = match spec {
            "@hourly" => "0 * * * *",
            "@daily" | "@midnight" => "0 0 * * *",
            "@weekly" => "0 0 * * 0",
            "@monthly" => "0 0 1 * *",
            s if s.starts_with("@every") => return Ok(Cron::Every(every(&s["@every".len()..])?)),
            s => s,
        };
        let parts: Vec<&str> = expanded.split_whitespace().collect();
        if parts.len() != 5 {
            bail!("cron needs five fields (min hour day-of-month month day-of-week) or @every/@hourly/@daily/@weekly/@monthly, got '{spec}'");
        }
        let (min, _) = field(parts[0], 0, 59, "minute")?;
        let (hour, _) = field(parts[1], 0, 23, "hour")?;
        let (dom, dom_any) = field(parts[2], 1, 31, "day-of-month")?;
        let (month, _) = field(parts[3], 1, 12, "month")?;
        let (mut dow, dow_any) = field(parts[4], 0, 7, "day-of-week")?;
        if dow & (1 << 7) != 0 {
            dow = (dow & !(1 << 7)) | 1;
        }
        Ok(Cron::Fields { min, hour, dom, month, dow, dom_any, dow_any })
    }

    /// The first firing strictly after `after` (µs since the epoch).
    pub(crate) fn next_after(&self, after: i64) -> Result<i64> {
        match self {
            Cron::Every(us) => Ok(after + us),
            Cron::Fields { min, hour, dom, month, dow, dom_any, dow_any } => {
                let start: DateTime<Utc> = Utc
                    .timestamp_micros(after)
                    .single()
                    .ok_or_else(|| miette!("cron: time out of range"))?;
                // next whole minute
                let mut t = start.with_second(0).unwrap().with_nanosecond(0).unwrap() + Duration::minutes(1);
                // at most five years of minutes, skipping whole hours and days
                let limit = t + Duration::days(366 * 5);
                while t < limit {
                    if month & (1 << t.month()) == 0 {
                        let (y, m) = if t.month() == 12 { (t.year() + 1, 1) } else { (t.year(), t.month() + 1) };
                        t = Utc.with_ymd_and_hms(y, m, 1, 0, 0, 0).unwrap();
                        continue;
                    }
                    let d_ok = dom & (1 << t.day()) != 0;
                    let w_ok = dow & (1 << t.weekday().num_days_from_sunday()) != 0;
                    let day_ok = match (dom_any, dow_any) {
                        (true, true) => true,
                        (true, false) => w_ok,
                        (false, true) => d_ok,
                        (false, false) => d_ok || w_ok,
                    };
                    if !day_ok {
                        t = (t + Duration::days(1)).with_hour(0).unwrap().with_minute(0).unwrap();
                        continue;
                    }
                    if hour & (1 << t.hour()) == 0 {
                        t = (t + Duration::hours(1)).with_minute(0).unwrap();
                        continue;
                    }
                    if min & (1 << t.minute()) == 0 {
                        t += Duration::minutes(1);
                        continue;
                    }
                    return Ok(t.timestamp_micros());
                }
                bail!("cron never fires")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(s: &str) -> i64 {
        DateTime::parse_from_rfc3339(s).unwrap().timestamp_micros()
    }

    #[test]
    fn fields_and_shorthands() {
        let c = Cron::parse("*/15 9-17 * * 1-5").unwrap();
        // Friday 2026-10-23 17:50 -> Monday 09:00
        assert_eq!(c.next_after(at("2026-10-23T17:50:00Z")).unwrap(), at("2026-10-26T09:00:00Z"));
        assert_eq!(c.next_after(at("2026-10-26T09:00:00Z")).unwrap(), at("2026-10-26T09:15:00Z"));
        let d = Cron::parse("@daily").unwrap();
        assert_eq!(d.next_after(at("2026-10-24T00:00:00Z")).unwrap(), at("2026-10-25T00:00:00Z"));
        let e = Cron::parse("@every 90s").unwrap();
        assert_eq!(e.next_after(0).unwrap(), 90_000_000);
        // Sunday as 7; day-of-month OR day-of-week when both are restricted
        let s = Cron::parse("0 12 13 * 7").unwrap();
        assert_eq!(s.next_after(at("2026-10-24T00:00:00Z")).unwrap(), at("2026-10-25T12:00:00Z"));
        assert_eq!(s.next_after(at("2026-11-12T13:00:00Z")).unwrap(), at("2026-11-13T12:00:00Z"));
        let m = Cron::parse("0 0 31 2 *").unwrap();
        assert!(m.next_after(0).is_err());
        for bad in ["* * *", "61 * * * *", "*/0 * * * *", "@every 5x", "a * * * *"] {
            assert!(Cron::parse(bad).is_err(), "{bad}");
        }
    }
}
