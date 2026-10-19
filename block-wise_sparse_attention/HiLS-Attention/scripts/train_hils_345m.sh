#!/usr/bin/env bash
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" hils_345m "$@"
