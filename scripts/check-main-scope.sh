#!/usr/bin/env bash
# main never edits what hardware branches own: crates/velox-variant/{src,kernels} stay exactly as
# they were when the variant crate was introduced, so `git merge main` into a hardware branch can
# never conflict. Run on main; exits non-zero on violations.
#   scripts/check-main-scope.sh            check HEAD
#   scripts/check-main-scope.sh --staged   check the index (pre-commit hook)
set -euo pipefail
baseline=7c7f18c2a9573fdbfcc5f67f9d60f595c2fb14bd  # feat: velox-inferer architecture on main
owned=(crates/velox-variant/src crates/velox-variant/kernels)
if [ "${1:-}" = "--staged" ]; then
  bad=$(git diff --cached --name-only -- "${owned[@]}")
else
  bad=$(git diff --name-only "$baseline" HEAD -- "${owned[@]}")
fi
if [ -n "$bad" ]; then
  echo "main changes files that hardware branches own:" >&2
  echo "$bad" | sed 's/^/  /' >&2
  echo "Extend the variant contract through main-owned crates instead (docs/branching.md)." >&2
  exit 1
fi
echo "main scope ok"
