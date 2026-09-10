#!/usr/bin/env bash
# Launches N simulated edge nodes as background processes, each training
# on its own private data shard and talking to the aggregator over
# gRPC + RabbitMQ. Usage: ./run_node.sh <total_nodes>
set -euo pipefail
cd "$(dirname "$0")/../edge_node"

TOTAL_NODES="${1:-5}"
export AGGREGATOR_ADDR="${AGGREGATOR_ADDR:-localhost:50051}"
export RABBITMQ_URL="${RABBITMQ_URL:-amqp://guest:guest@localhost:5672/}"

mkdir -p /tmp/fedlearn_logs
for i in $(seq 0 $((TOTAL_NODES - 1))); do
  echo "Launching node-$i (logs: /tmp/fedlearn_logs/node-$i.log)"
  python3 -u node_client.py \
    --node-id "node-$i" --node-index "$i" --total-nodes "$TOTAL_NODES" \
    --local-epochs 4 --lr 0.15 --clip-norm 2.0 --noise-multiplier 0.2 \
    > "/tmp/fedlearn_logs/node-$i.log" 2>&1 &
done

echo "Launched $TOTAL_NODES nodes. Tail logs with:"
echo "  tail -f /tmp/fedlearn_logs/node-*.log"
wait
