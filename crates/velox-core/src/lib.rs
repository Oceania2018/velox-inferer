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
pub mod ggml;
pub mod kv;
pub mod model;
pub mod op;
pub mod plan;
pub mod profile;
pub mod quant;
pub mod rng;
pub mod variant;

pub use backend::{
    ActPolicy, ActQuant, AttnPrecision, AttnShape, Backend, BackendKind, ConvShape, CopySpec,
    DilatedConvShape, ExpertBank, ExpertMatrix, FullAttnShape, GdnReplayLayer, GdnShape, GeluKind,
    HcCombineShape, KvStoreShape, LinearDst, LinearPart, MoeRouteShape, NormKind, QsaPoolShape,
    QsaSelectShape, RopeLayout, RopePosShape, RopeShape, RowGateShape, WeightView, GRAPH_KV_BUCKET,
    GRAPH_STATIC,
};
pub use error::{Error, Result};
pub use format::WeightFormat;
pub use kv::KvFormat;
pub use op::Op;
pub use plan::{MemoryPlan, ModelSpec, PlanPolicy, PlanRequest};
pub use profile::{DeviceProfile, FormFactor, GpuArch, HwCaps, Vendor};
pub use variant::VariantInfo;
