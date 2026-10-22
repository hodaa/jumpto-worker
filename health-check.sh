#!/bin/bash
if [ "$(docker inspect -f '{{.State.Running}}' qfza-worker)" != "true" ]; then
    echo "Worker is not running"
    exit 1
fi