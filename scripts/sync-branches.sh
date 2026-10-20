#!/usr/bin/env bash
# Merges main into every hardware branch. main wins: conflicting hunks resolve to main's side,
# and any file a branch changed outside crates/velox-variant/{src,kernels} is reset to main's
# version, so a branch can never carry a design that diverges from main. Branch-owned files
# cannot conflict because main never edits them (scripts/check-main-scope.sh).
#   scripts/sync-branches.sh [branch ...]   (default: every hardware branch)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ $# -gt 0 ]; then branches=("$@"); else branches=(rtx-5090-32g rtx-5090-24g rtx-5060-ti rtx-4090); fi
bash scripts/check-main-scope.sh >/dev/null
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for b in "${branches[@]}"; do
  # A branch checked out elsewhere (e.g. a build worktree) is merged there.
  dir="$(git worktree list --porcelain | awk -v b="refs/heads/$b" '/^worktree /{w=$2} $0=="branch "b{print w}')"
  if [ -z "$dir" ]; then
    dir="$tmp/$b"
    git worktree add -q "$dir" "$b"
  fi
  git -C "$dir" merge -q --no-edit -X theirs main
  drift=$(git -C "$dir" diff --name-only main HEAD | grep -v -E '^crates/velox-variant/(src|kernels)/' || true)
  if [ -n "$drift" ]; then
    echo "$b: resetting files outside its scope to main:" && echo "$drift" | sed 's/^/  /'
    while read -r f; do
      if git cat-file -e "main:$f" 2>/dev/null; then git -C "$dir" checkout main -- "$f"; else git -C "$dir" rm -q -- "$f"; fi
    done <<< "$drift"
    git -C "$dir" commit -q -m "chore($b): reset out-of-scope files to main" -m "main is authoritative outside crates/velox-variant/{src,kernels} (docs/branching.md)."
  fi
  (cd "$dir" && bash scripts/check-branch-scope.sh main)
  echo "$b: merged main at $(git rev-parse --short main)"
  case "$dir" in "$tmp"/*) git worktree remove --force "$dir" ;; esac
done
