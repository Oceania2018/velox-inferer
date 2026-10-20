//! Device profiles: the hardware catalog shared by every branch.
//!
//! A profile describes one GPU SKU (a 32 GB desktop RTX 5090 and a 24 GB RTX 5090 Laptop GPU are
//! different profiles). Profiles drive memory planning and tell the user which hardware branch
//! carries tuned kernels for their card. Adding a card is a catalog entry on main; tuning it is
//! work on that card's branch.

use std::fmt;

use crate::backend::BackendKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vendor {
    Nvidia,
    Amd,
}

/// Instruction-set generation, which decides which compiled kernels can run on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuArch {
    /// NVIDIA compute capability, e.g. 8.9 (Ada) or 12.0 (consumer Blackwell).
    Cuda { major: u32, minor: u32 },
    /// AMD GPU target, e.g. "gfx1100" (RDNA3).
    Hip { gfx: &'static str },
}

impl GpuArch {
    pub fn vendor(&self) -> Vendor {
        match self {
            GpuArch::Cuda { .. } => Vendor::Nvidia,
            GpuArch::Hip { .. } => Vendor::Amd,
        }
    }

    pub fn backend(&self) -> BackendKind {
        match self {
            GpuArch::Cuda { .. } => BackendKind::Cuda,
            GpuArch::Hip { .. } => BackendKind::Hip,
        }
    }

    /// Hardware number formats and memory engines available on this architecture.
    pub fn caps(&self) -> HwCaps {
        match *self {
            GpuArch::Cuda { major, minor } => {
                let cc = major * 10 + minor;
                HwCaps {
                    bf16: cc >= 80,
                    fp8: cc >= 89,
                    fp4: cc >= 100,
                    tma: cc >= 90,
                }
            }
            GpuArch::Hip { gfx } => HwCaps {
                bf16: gfx.starts_with("gfx11")
                    || gfx.starts_with("gfx12")
                    || gfx.starts_with("gfx9"),
                fp8: gfx.starts_with("gfx12") || gfx == "gfx942",
                fp4: false,
                tma: false,
            },
        }
    }
}

impl fmt::Display for GpuArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuArch::Cuda { major, minor } => write!(f, "sm_{major}{minor}"),
            GpuArch::Hip { gfx } => write!(f, "{gfx}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HwCaps {
    pub bf16: bool,
    pub fp8: bool,
    pub fp4: bool,
    /// Tensor Memory Accelerator (bulk async copies), sm_90+.
    pub tma: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormFactor {
    Desktop,
    Laptop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceProfile {
    /// Stable id used on the command line and in VELOX_PROFILE, e.g. "rtx-5090-laptop".
    pub id: &'static str,
    pub name: &'static str,
    pub arch: GpuArch,
    pub form_factor: FormFactor,
    pub vram_gib: u32,
    /// Streaming multiprocessors (NVIDIA) or compute units (AMD).
    pub sm_count: u32,
    /// Peak DRAM bandwidth in GB/s; decode speed is bounded by it.
    pub mem_bandwidth_gbs: u32,
    /// VRAM kept free for the driver, display and fragmentation unless the variant overrides it.
    pub reserved_mib: u32,
    /// Branch carrying tuned kernels for this card; None when only portable kernels exist.
    pub tuned_branch: Option<&'static str>,
}

impl DeviceProfile {
    pub fn vendor(&self) -> Vendor {
        self.arch.vendor()
    }

    pub fn vram_bytes(&self) -> u64 {
        u64::from(self.vram_gib) << 30
    }
}

// Figures are from the vendors' public specification pages. Bandwidth and SM counts are used for
// planning and for reporting achieved bandwidth, never for correctness.
pub static CATALOG: &[DeviceProfile] = &[
    DeviceProfile {
        id: "rtx-5090",
        name: "NVIDIA GeForce RTX 5090",
        arch: GpuArch::Cuda {
            major: 12,
            minor: 0,
        },
        form_factor: FormFactor::Desktop,
        vram_gib: 32,
        sm_count: 170,
        mem_bandwidth_gbs: 1792,
        reserved_mib: 1024,
        tuned_branch: Some("rtx-5090-32g"),
    },
    DeviceProfile {
        id: "rtx-5090-laptop",
        name: "NVIDIA GeForce RTX 5090 Laptop GPU",
        arch: GpuArch::Cuda {
            major: 12,
            minor: 0,
        },
        form_factor: FormFactor::Laptop,
        vram_gib: 24,
        sm_count: 82,
        mem_bandwidth_gbs: 896,
        reserved_mib: 1536,
        tuned_branch: Some("rtx-5090-24g"),
    },
    DeviceProfile {
        id: "rtx-5060-ti-16g",
        name: "NVIDIA GeForce RTX 5060 Ti 16GB",
        arch: GpuArch::Cuda {
            major: 12,
            minor: 0,
        },
        form_factor: FormFactor::Desktop,
        vram_gib: 16,
        sm_count: 36,
        mem_bandwidth_gbs: 448,
        reserved_mib: 768,
        tuned_branch: Some("rtx-5060-ti"),
    },
    DeviceProfile {
        id: "rtx-5060-ti-8g",
        name: "NVIDIA GeForce RTX 5060 Ti 8GB",
        arch: GpuArch::Cuda {
            major: 12,
            minor: 0,
        },
        form_factor: FormFactor::Desktop,
        vram_gib: 8,
        sm_count: 36,
        mem_bandwidth_gbs: 448,
        reserved_mib: 640,
        tuned_branch: Some("rtx-5060-ti"),
    },
    DeviceProfile {
        id: "rtx-4090",
        name: "NVIDIA GeForce RTX 4090",
        arch: GpuArch::Cuda { major: 8, minor: 9 },
        form_factor: FormFactor::Desktop,
        vram_gib: 24,
        sm_count: 128,
        mem_bandwidth_gbs: 1008,
        reserved_mib: 1024,
        tuned_branch: Some("rtx-4090"),
    },
    // AMD is catalogued so planning works today; it needs the HIP backend (docs/adding-amd.md).
    DeviceProfile {
        id: "rx-7900-xtx",
        name: "AMD Radeon RX 7900 XTX",
        arch: GpuArch::Hip { gfx: "gfx1100" },
        form_factor: FormFactor::Desktop,
        vram_gib: 24,
        sm_count: 96,
        mem_bandwidth_gbs: 960,
        reserved_mib: 1024,
        tuned_branch: None,
    },
];

pub fn by_id(id: &str) -> Option<&'static DeviceProfile> {
    CATALOG.iter().find(|p| p.id.eq_ignore_ascii_case(id))
}

/// Matches a physical device to a profile by architecture and SM count, then by the nearest VRAM
/// size (SKUs such as the 8 GB and 16 GB RTX 5060 Ti share a die). Device names are not used:
/// they vary across drivers and OEMs.
pub fn detect(
    arch: GpuArch,
    sm_count: u32,
    total_mem_bytes: u64,
) -> Option<&'static DeviceProfile> {
    let total_mib = (total_mem_bytes >> 20) as i64;
    CATALOG
        .iter()
        .filter(|p| p.arch == arch && p.sm_count == sm_count)
        .min_by_key(|p| (i64::from(p.vram_gib) * 1024 - total_mib).abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SM120: GpuArch = GpuArch::Cuda {
        major: 12,
        minor: 0,
    };

    #[test]
    fn ids_are_unique() {
        for (i, a) in CATALOG.iter().enumerate() {
            assert!(
                CATALOG[i + 1..].iter().all(|b| b.id != a.id),
                "duplicate id {}",
                a.id
            );
        }
    }

    #[test]
    fn detects_desktop_and_laptop_5090() {
        // Reported totals are slightly below the nominal size.
        assert_eq!(detect(SM120, 170, 32607 << 20).unwrap().id, "rtx-5090");
        assert_eq!(
            detect(SM120, 82, 24463 << 20).unwrap().id,
            "rtx-5090-laptop"
        );
    }

    #[test]
    fn detects_5060_ti_by_vram() {
        assert_eq!(
            detect(SM120, 36, 16311 << 20).unwrap().id,
            "rtx-5060-ti-16g"
        );
        assert_eq!(detect(SM120, 36, 8151 << 20).unwrap().id, "rtx-5060-ti-8g");
    }

    #[test]
    fn unknown_device_is_none() {
        assert!(detect(GpuArch::Cuda { major: 8, minor: 6 }, 84, 24 << 30).is_none());
    }

    #[test]
    fn caps_follow_arch() {
        let ada = by_id("rtx-4090").unwrap().arch.caps();
        assert!(ada.fp8 && !ada.fp4 && !ada.tma);
        let bw = by_id("rtx-5090").unwrap().arch.caps();
        assert!(bw.fp8 && bw.fp4 && bw.tma);
        assert_eq!(
            by_id("rx-7900-xtx").unwrap().arch.backend(),
            BackendKind::Hip
        );
    }
}
