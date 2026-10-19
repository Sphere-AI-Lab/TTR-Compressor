#!/usr/bin/env bash
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" ttr_conv_345m "$@"
