/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use candle_core::Device;

use crate::Result;

/// Where to run a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DevicePref {
    /// Apple GPU when built with `metal` and one is present, else CPU.
    Auto,
    /// CPU.
    Cpu,
    /// Apple GPU (error when unavailable).
    Metal,
}

impl DevicePref {
    /// Parse `auto`, `cpu`, `metal`/`gpu`.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "auto" => DevicePref::Auto,
            "cpu" => DevicePref::Cpu,
            "metal" | "gpu" | "mps" => DevicePref::Metal,
            o => return Err(crate::Error::Config(format!("unknown device '{o}'"))),
        })
    }

    pub(crate) fn device(self) -> Result<Device> {
        match self {
            DevicePref::Cpu => Ok(Device::Cpu),
            DevicePref::Metal => Ok(Device::new_metal(0)?),
            DevicePref::Auto => {
                #[cfg(feature = "metal")]
                if let Ok(d) = Device::new_metal(0) {
                    return Ok(d);
                }
                Ok(Device::Cpu)
            }
        }
    }
}

pub(crate) fn device_name(d: &Device) -> &'static str {
    match d {
        Device::Cpu => "cpu",
        Device::Metal(_) => "metal",
        Device::Cuda(_) => "cuda",
    }
}
