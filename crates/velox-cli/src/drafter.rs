//! The block drafter speculative decoding uses: `--drafter dflash2|dspine` (default dflash2),
//! `--draft-artifact <.vi>` (where its component lives; default the model artifact) and
//! `--draft-window <n>` (context rows it attends to, at most the trained window; fewer rows shrink
//! every lane's drafter rings; 0: the trained window) and `--draft-vocab full|draft` (DSpine:
//! propose over the target's full output head, the default, or its draft vocabulary head).

use velox_core::{Backend, Error, Result};
use velox_models::qwen3_5::{DFlashModel, DraftKind};
use velox_weights::Artifact;

use crate::args::Args;

pub struct DraftSource {
    pub kind: DraftKind,
    /// DSpine was asked for: a missing component is an error (a missing DFlash2 falls back to MTP).
    explicit: bool,
    art: Option<Artifact>,
    window: Option<usize>,
    full_head: bool,
}

impl DraftSource {
    pub fn from_args(a: &Args) -> Result<Self> {
        let kind = DraftKind::parse(a.get("drafter").unwrap_or("dflash2"))?;
        let art = match a.get("draft-artifact") {
            Some(p) => Some(Artifact::open(p).map_err(|e| Error::Config(format!("{p}: {e}")))?),
            None => None,
        };
        let window = match a.parse("draft-window", 0usize)? {
            0 => None,
            n => Some(n),
        };
        let full_head = match a.get("draft-vocab").unwrap_or("full") {
            "draft" => false,
            "full" => true,
            other => {
                return Err(Error::Config(format!(
                    "--draft-vocab must be full or draft, got '{other}'"
                )))
            }
        };
        Ok(DraftSource {
            kind,
            explicit: kind == DraftKind::DSpine,
            art,
            window,
            full_head,
        })
    }

    fn artifact<'a>(&'a self, main: &'a Artifact) -> &'a Artifact {
        self.art.as_ref().unwrap_or(main)
    }

    /// Whether the drafter's component is there; an error when it was asked for and is not.
    pub fn available(&self, main: &Artifact) -> Result<bool> {
        let has = self
            .artifact(main)
            .components()
            .contains_key(self.kind.component());
        if !has && self.explicit {
            return Err(Error::Config(format!(
                "--drafter {}: {} has no {} component",
                self.kind.component(),
                self.artifact(main).path.display(),
                self.kind.component()
            )));
        }
        Ok(has)
    }

    pub fn load<B: Backend>(
        &self,
        b: &B,
        main: &Artifact,
        hidden: usize,
    ) -> Result<Option<DFlashModel<B>>> {
        let mut dm =
            DFlashModel::load_kind(b, self.artifact(main), hidden, self.kind, self.window)?;
        if let Some(d) = dm.as_mut() {
            d.set_full_head(self.full_head);
        }
        Ok(dm)
    }
}
