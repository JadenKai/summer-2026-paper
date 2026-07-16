#!/usr/bin/env bash
# NixOS: expose libstdc++ from nix-ld so numpy/pandas native extensions load.
NIX_LD_LIBS=/run/current-system/sw/share/nix-ld/lib
export LD_LIBRARY_PATH="${NIX_LD_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec uv run "$@"
