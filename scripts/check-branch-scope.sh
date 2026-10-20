#!/usr/bin/env bash
# Hardware branches may only change crates/velox-variant/ (except its Cargo.toml and build.rs,
# which must stay identical to main). Run on a hardware branch; exits non-zero on violations.
#   scripts/check-branch-scope.sh [base-ref]   (default: main)
set -euo pipefail
base="${1:-main}"
branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" = "$base" ]; then
  echo "on $base: nothing to check"; exit 0
fi
bad=$(git diff --name-only "$(git merge-base "$base" HEAD)" HEAD | grep -v -E '^crates/velox-variant/(src|kernels)/' || true)
if [ -n "$bad" ]; then
  echo "branch '$branch' changes files outside crates/velox-variant/{src,kernels}:" >&2
  echo "$bad" | sed 's/^/  /' >&2
  echo "Move shared work to $base (docs/branching.md)." >&2
  exit 1
fi
echo "branch '$branch' scope ok"
