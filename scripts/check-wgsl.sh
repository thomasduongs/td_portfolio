#!/usr/bin/env bash
# Validate WGSL shaders with naga, prepending the shared frame.wgsl like the engine does.
# Usage: scripts/check-wgsl.sh [file.wgsl ...]   (default: all shaders except frame.wgsl; beams.wgsl is auto-prepended for files that call beams()/beamI()/letterMask())
set -u
NAGA="${NAGA:-$HOME/.cargo/bin/naga}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/src/engine/shaders"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
files=("$@"); [ ${#files[@]} -eq 0 ] && files=($(ls "$DIR"/*.wgsl | grep -v '/frame.wgsl$'))
fail=0
for f in "${files[@]}"; do
  name="$(basename "$f")"
  if [ "$name" = "beams.wgsl" ]; then continue; fi
  if [ "$name" = "jfa.wgsl" ] && ! grep -q "F\." "$f"; then cp "$f" "$TMP/$name";
  elif grep -qE "beams\(|beamI\(|letterMask\(|EXCITE_COLOR" "$f"; then cat "$DIR/frame.wgsl" "$DIR/beams.wgsl" "$f" > "$TMP/$name";  # beam consumers
  else cat "$DIR/frame.wgsl" "$f" > "$TMP/$name"; fi
  if out="$("$NAGA" "$TMP/$name" 2>&1)"; then echo "OK   $name"; else echo "FAIL $name"; echo "$out" | head -30; fail=1; fi
done
exit $fail
