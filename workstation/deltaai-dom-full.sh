#!/usr/bin/env bash
# The DOM baseline moved to Delta A100; preserve this command as an alias.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/delta-dom-full.sh" "$@"
