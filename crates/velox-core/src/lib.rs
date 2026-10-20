//! Hardware-agnostic core of velox-inferer.
//!
//! Everything here is shared by every hardware branch and must stay free of vendor code:
//! device profiles (which cards exist and what they offer), the [`Backend`] op contract,
//! the VRAM memory planner, the variant description, and a reference transformer model
//! written against [`Backend`].

pub mod backend;
pub mod check;
pub mod error;
pub mod format;
pub mod model;
pub mod op;
pub mod plan;
pub mod profile;
pub mod rng;
pub mod variant;

pub use backend::{
    AttnShape, Backend, BackendKind, ConvShape, FullAttnShape, GdnShape, GeluKind, NormKind,
    RopeLayout, RopePosShape, RopeShape, WeightView,
};
pub use error::{Error, Result};
pub use format::WeightFormat;
pub use op::Op;
pub use plan::{MemoryPlan, ModelSpec, PlanPolicy, PlanRequest};
pub use profile::{DeviceProfile, FormFactor, GpuArch, HwCaps, Vendor};
pub use variant::VariantInfo;
