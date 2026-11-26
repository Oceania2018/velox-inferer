//! What this binary was built from (see `build.rs`): served at `GET /version`, as
//! `velox_build_info` in `/metrics` and in the startup line, so a node's deployment can be checked
//! against its branch (`git ls-remote origin <ref>`).

use serde_json::{json, Value};

/// Crate version (the workspace's `version`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Full commit SHA, or `unknown`.
pub const COMMIT: &str = env!("VELOX_BUILD_SHA");
/// The branch, tag or SHA the build was asked for; empty when not known.
pub const REF: &str = env!("VELOX_BUILD_REF");
/// Whether a local build had uncommitted changes.
pub const DIRTY: bool = matches!(env!("VELOX_BUILD_DIRTY").as_bytes(), b"true");
/// Build time, Unix seconds.
pub const BUILD_UNIX: &str = env!("VELOX_BUILD_TIME_UNIX");
/// The repository's change log as of this build (`GET /changes`).
pub const CHANGES: &str = include_str!("../../../CHANGES.md");
/// Entries of [`CHANGES`] that `GET /version` lists.
const RECENT_CHANGES: usize = 5;

/// One `## date · #PR · title` entry of [`CHANGES`].
pub struct Change {
    pub date: &'static str,
    pub prs: &'static str,
    pub title: &'static str,
    pub body: String,
}

/// The entries of [`CHANGES`], newest first (the file's order).
pub fn changes() -> Vec<Change> {
    let mut out: Vec<Change> = Vec::new();
    for line in CHANGES.lines() {
        if let Some(head) = line.strip_prefix("## ") {
            let mut parts = head.splitn(3, " · ");
            let (a, b, c) = (parts.next(), parts.next(), parts.next());
            let (date, prs, title) = match (a, b, c) {
                (Some(d), Some(p), Some(t)) => (d.trim(), p.trim(), t.trim()),
                _ => ("", "", head.trim()),
            };
            out.push(Change {
                date,
                prs,
                title,
                body: String::new(),
            });
        } else if let Some(c) = out.last_mut() {
            let l = line.trim();
            if !l.is_empty() {
                if !c.body.is_empty() {
                    c.body.push(' ');
                }
                c.body.push_str(l);
            }
        }
    }
    out
}

/// The commit's first 7 characters (all of it when shorter, e.g. `unknown`).
pub fn commit_short() -> &'static str {
    COMMIT
        .get(..7)
        .filter(|_| COMMIT != "unknown")
        .unwrap_or(COMMIT)
}

fn build_unix() -> u64 {
    BUILD_UNIX.parse().unwrap_or(0)
}

/// `0.1.0+c99a79c` (`-dirty` appended for uncommitted changes).
pub fn version_string() -> String {
    format!(
        "{VERSION}+{}{}",
        commit_short(),
        if DIRTY { "-dirty" } else { "" }
    )
}

/// One line for logs: `velox 0.1.0+c99a79c (rtx-5090-32g, built 2026-11-26T00:10:00Z)`.
pub fn describe() -> String {
    let r = if REF.is_empty() { "ref unknown" } else { REF };
    format!(
        "velox {} ({r}, built {})",
        version_string(),
        rfc3339(build_unix())
    )
}

/// The `GET /version` body.
pub fn json() -> Value {
    json!({
        "name": "velox-inferer",
        "version": version_string(),
        "package_version": VERSION,
        "commit": COMMIT,
        "commit_short": commit_short(),
        "ref": REF,
        "dirty": DIRTY,
        "built_at": rfc3339(build_unix()),
        "built_unix": build_unix(),
        "recent_changes": changes()
            .into_iter()
            .take(RECENT_CHANGES)
            .map(|c| json!({"date": c.date, "prs": c.prs, "title": c.title, "summary": c.body}))
            .collect::<Vec<_>>(),
        "changes_url": "/changes",
    })
}

/// `velox_build_info{...} 1` for `/metrics`.
pub fn metric() -> String {
    format!(
        "# HELP velox_build_info Build of this binary: version, commit and the ref it was built from\n\
         # TYPE velox_build_info gauge\n\
         velox_build_info{{version=\"{}\",commit=\"{COMMIT}\",ref=\"{}\",built_unix=\"{}\"}} 1\n",
        version_string(),
        REF.replace('\\', "\\\\").replace('"', "\\\""),
        build_unix()
    )
}

/// UTC `YYYY-MM-DDTHH:MM:SSZ` from Unix seconds (days-to-civil, no date crate).
fn rfc3339(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3_600,
        rem % 3_600 / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_known_dates() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(1_795_650_915), "2026-11-25T23:55:15Z");
    }

    #[test]
    fn version_json_has_the_commit() {
        let v = json();
        assert_eq!(v["commit"], COMMIT);
        assert!(v["version"].as_str().unwrap().starts_with(VERSION));
        assert!(metric().contains("velox_build_info{"));
    }

    #[test]
    fn change_log_parses_newest_first() {
        let c = changes();
        assert!(c.len() >= 2, "CHANGES.md has entries");
        for e in &c {
            // `## YYYY-MM-DD · #PR · title`
            assert_eq!(e.date.len(), 10, "date of {:?}", e.title);
            assert!(e.prs.starts_with('#'), "PR of {:?}", e.title);
            assert!(!e.title.is_empty());
        }
        assert!(c.windows(2).all(|w| w[0].date >= w[1].date), "newest first");
        assert!(json()["recent_changes"].as_array().unwrap().len() <= RECENT_CHANGES);
    }
}
