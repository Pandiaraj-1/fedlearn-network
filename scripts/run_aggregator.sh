#!/usr/bin/env bash
# Builds and runs the central aggregator directly (no Docker).
# Requires: RabbitMQ already running and reachable, Go 1.22+.
set -euo pipefail
cd "$(dirname "$0")/../aggregator"

export MIN_NODES_PER_ROUND="${MIN_NODES_PER_ROUND:-5}"
export MAX_ROUNDS="${MAX_ROUNDS:-25}"
export RABBITMQ_URL="${RABBITMQ_URL:-amqp://guest:guest@localhost:5672/}"
export GRPC_ADDR="${GRPC_ADDR:-:50051}"
export METRICS_ADDR="${METRICS_ADDR:-:9100}"

go build -o /tmp/aggregator .
echo "Starting aggregator: gRPC=$GRPC_ADDR metrics=$METRICS_ADDR min_nodes=$MIN_NODES_PER_ROUND max_rounds=$MAX_ROUNDS"
exec /tmp/aggregator
