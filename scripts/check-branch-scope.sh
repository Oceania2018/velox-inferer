#!/usr/bin/env bash
# Hardware branches may only change crates/velox-variant/{src,kernels}; everything else, including
# that crate's Cargo.toml and build.rs and Cargo.lock, stays identical to main. Run on a hardware
# branch; exits non-zero on violations.
#   scripts/check-branch-scope.sh [base-ref]   whole branch against its merge base (default main)
#   scripts/check-branch-scope.sh --staged     the index only (pre-commit hook)
set -euo pipefail
allowed='^crates/velox-variant/(src|kernels)/'
if [ "${1:-}" = "--staged" ]; then
  bad=$(git diff --cached --name-only | grep -v -E "$allowed" || true)
else
  base="${1:-main}"
  branch="$(git rev-parse --abbrev-ref HEAD)"
  if [ "$branch" = "$base" ]; then
    echo "on $base: nothing to check"; exit 0
  fi
  bad=$(git diff --name-only "$(git merge-base "$base" HEAD)" HEAD | grep -v -E "$allowed" || true)
fi
if [ -n "$bad" ]; then
  echo "hardware branch changes files outside crates/velox-variant/{src,kernels}:" >&2
  echo "$bad" | sed 's/^/  /' >&2
  echo "Move shared work to main (docs/branching.md)." >&2
  exit 1
fi
echo "branch scope ok"
