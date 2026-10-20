//! Description of the hardware variant compiled into this build.
//!
//! Exactly one variant exists per build: the `velox-variant` crate of the checked-out branch.
//! main carries the generic variant (portable kernels, any device); each hardware branch replaces
//! that crate with kernels and policies tuned for its cards.

use crate::profile::{self, DeviceProfile, GpuArch};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VariantInfo {
    pub id: &'static str,
    pub branch: &'static str,
    pub description: &'static str,
    /// Profile ids this variant is tuned for. Empty means generic: no device is preferred.
    pub profiles: &'static [&'static str],
    /// Architectures the variant's compiled kernels can run on. Empty means any (portable only).
    pub archs: &'static [GpuArch],
}

impl VariantInfo {
    pub fn is_generic(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn tunes(&self, p: &DeviceProfile) -> bool {
        self.profiles.iter().any(|id| id.eq_ignore_ascii_case(p.id))
    }

    pub fn runs_on(&self, arch: GpuArch) -> bool {
        self.archs.is_empty() || self.archs.contains(&arch)
    }

    /// One line for the user when this build is not the best one for `p`, naming the branch to
    /// check out instead. None when the build already matches.
    pub fn advice(&self, p: &DeviceProfile) -> Option<String> {
        if self.tunes(p) {
            return None;
        }
        match p.tuned_branch {
            Some(branch) if branch != self.branch => Some(format!(
                "{} has tuned kernels on branch '{branch}'; this build is '{}' ({}). \
                 Check out '{branch}' for best performance.",
                p.id, self.branch, self.id
            )),
            Some(_) => None,
            None => Some(format!(
                "{} has no tuned branch yet; running portable kernels.",
                p.id
            )),
        }
    }

    /// Profiles the variant names that are missing from the catalog (a branch bug).
    pub fn unknown_profiles(&self) -> Vec<&'static str> {
        self.profiles
            .iter()
            .copied()
            .filter(|id| profile::by_id(id).is_none())
            .collect()
    }
}
