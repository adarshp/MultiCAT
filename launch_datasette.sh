#!/usr/bin/env bash
# Shell script for launching a Datasette instance with the MultiCAT database

set -euo pipefail
datasette -i \
    multicat.db \
    --metadata metadata.yml \
