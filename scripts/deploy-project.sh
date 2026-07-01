#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <deployment-slug> [wrangler args...]" >&2
  exit 1
fi

slug="$1"
shift
config="deployments/${slug}/wrangler.jsonc"

if [ ! -f "$config" ]; then
  echo "No deployment config found at ${config}" >&2
  exit 1
fi

npx -y wrangler@latest deploy --config "$config" "$@"
