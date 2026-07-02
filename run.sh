#!/usr/bin/env bash
# NixOS: expose libstdc++ from nix-ld so numpy/pandas native extensions load.
NIX_LD_LIBS=/nix/store/4v0n0xymnx2l2q2jg7hn60bfnjkcfqgp-ld-library-path/share/nix-ld/lib
export LD_LIBRARY_PATH="${NIX_LD_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec uv run "$@"
