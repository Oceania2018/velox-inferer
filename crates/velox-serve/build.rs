//! Build identity served at `GET /version` and in `/metrics` (`velox_build_info`): which commit and
//! ref this binary was built from, so a deployment can be checked against its branch.
//!
//! Sources, first found wins:
//! - `VELOX_GIT_SHA` / `VELOX_GIT_REF` / `VELOX_BUILD_UNIX` from the environment (the Docker build
//!   sets them; a git-URL build context has no `.git` unless kept with
//!   `BUILDKIT_CONTEXT_KEEP_GIT_DIR=1`);
//! - `git` on the checkout (local builds), with a dirty flag for uncommitted changes;
//! - otherwise `unknown`.

use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn env(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={name}");
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn main() {
    let from_env = env("VELOX_GIT_SHA");
    let sha = from_env
        .clone()
        .or_else(|| git(&["rev-parse", "HEAD"]))
        .unwrap_or_else(|| "unknown".into());
    let git_ref = env("VELOX_GIT_REF")
        .or_else(|| git(&["rev-parse", "--abbrev-ref", "HEAD"]).filter(|r| r != "HEAD"))
        .unwrap_or_default();
    // Uncommitted changes only mean something for a local checkout.
    let dirty =
        from_env.is_none() && git(&["status", "--porcelain", "--untracked-files=no"]).is_some();
    let built = env("VELOX_BUILD_UNIX").unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
            .to_string()
    });

    // A local checkout rebuilds when HEAD moves (new commit, checkout) — not on every build.
    if from_env.is_none() {
        for (what, path) in [
            ("--git-path", "HEAD"),
            ("--git-path", "packed-refs"),
            ("--git-path", "index"),
        ] {
            if let Some(p) = git(&["rev-parse", what, path]) {
                if Path::new(&p).exists() {
                    println!("cargo:rerun-if-changed={p}");
                }
            }
        }
        if let Some(full) = git(&["rev-parse", "--symbolic-full-name", "HEAD"]) {
            if let Some(p) = git(&["rev-parse", "--git-path", &full]) {
                if Path::new(&p).exists() {
                    println!("cargo:rerun-if-changed={p}");
                }
            }
        }
    }

    println!("cargo:rustc-env=VELOX_BUILD_SHA={sha}");
    println!("cargo:rustc-env=VELOX_BUILD_REF={git_ref}");
    println!("cargo:rustc-env=VELOX_BUILD_DIRTY={dirty}");
    println!("cargo:rustc-env=VELOX_BUILD_TIME_UNIX={built}");
}
