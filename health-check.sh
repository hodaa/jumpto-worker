if [ "$(docker inspect -f '{{.State.Running}}' jumpto-worker)" != "true" ]; then
    echo "Worker is not running"
    ...
fi