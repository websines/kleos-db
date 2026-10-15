/*
 * Copyright 2022, The Cozo Project Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/// Return early with a [`miette::Report`]; same arguments as `miette::bail!`.
///
/// miette's own `bail!` expands to `return Err(..);`, and that trailing semicolon is
/// rejected in expression position (e.g. `Err(e) => bail!(e),`) by rustc's
/// `semicolon_in_expressions_from_non_local_macros` future-compat lint. This expansion
/// has no semicolon, so it is valid both as a statement and as an expression.
/// Import it as `use crate::bail;`.
macro_rules! bail {
    ($($arg:tt)*) => {
        return ::core::result::Result::Err(::miette::miette!($($arg)*))
    };
}
pub(crate) use bail;

#[inline(always)]
pub(crate) fn swap_option_result<T, E>(d: Result<Option<T>, E>) -> Option<Result<T, E>> {
    match d {
        Ok(Some(s)) => Some(Ok(s)),
        Ok(None) => None,
        Err(e) => Some(Err(e)),
    }
}

#[derive(Default)]
pub(crate) struct TempCollector<T: serde::Serialize + for<'a> serde::Deserialize<'a>> {
    // pub(crate) inner: Vec<T>,
    pub(crate) inner: swapvec::SwapVec<T>,
}

impl<T: serde::Serialize + for<'a> serde::Deserialize<'a>> TempCollector<T> {
    pub(crate) fn push(&mut self, val: T) {
        self.inner.push(val).unwrap();
    }
    pub(crate) fn into_iter(self) -> impl Iterator<Item = T> {
        self.inner.into_iter().map(|v| v.unwrap())
    }
}
