# Privacy-Preserving Federated Learning Network

A decentralized training system where multiple edge nodes train a shared
model locally and send only cryptographically-bounded, differentially-private
model weights to a central aggregator — raw data never leaves a node.

This is a real, runnable implementation. Every component below was built,
compiled, and run end-to-end (5 simulated edge nodes, 25 training rounds)
during development; see `docs/sample_run_aggregator.log` and
`docs/sample_run_node-*.log` for the actual output of that run.

## Architecture

```
                    ┌─────────────────────────┐
                    │   Central Aggregator     │
                    │   (Go / gRPC)             │
                    │                          │
   ┌──────────┐     │  • RegisterNode          │      ┌──────────────┐
   │ Edge Node│────▶│  • GetGlobalModel        │◀────▶│  Prometheus  │
   │ (Node 0) │ gRPC│  • FedAvg aggregation     │      │  /metrics    │
   └────┬─────┘     │                          │      └──────────────┘
        │           └───────────▲──────────────┘
        │ publish               │ consume
        ▼           ┌───────────┴──────────────┐
   ┌──────────┐     │        RabbitMQ            │
   │ Edge Node│────▶│      "model_updates"       │
   │ (Node N) │     │           queue             │
   └──────────┘     └────────────────────────────┘
```

| Component | Technology | Role |
|---|---|---|
| Central Aggregator | Go / gRPC | Coordinates training rounds and merges model weights using Federated Averaging (FedAvg). |
| Edge Nodes | Python / PyTorch | Simulate client devices, each training on an isolated, private data shard. |
| Privacy Layer | DP-SGD (gradient clipping + Gaussian noise) | Prevents reverse-engineering of transmitted updates. See "About the privacy layer" below for why this replaces a direct PySyft dependency. |
| Message Broker | RabbitMQ | Handles asynchronous, decoupled publishing of updates from many distributed nodes. |
| Observability | Prometheus | Exposes round number, global loss, node counts, and aggregation latency. |

### How a round works
1. An edge node calls `GetGlobalModel` (gRPC) to fetch the current global
   weights and round number.
2. It trains locally for a few epochs on its own private data — this data
   never leaves the process.
3. After every local training step, gradients are **clipped to a bounded
   L2 norm and injected with Gaussian noise** (the privacy layer).
4. The node publishes the resulting weight vector to RabbitMQ's
   `model_updates` queue — never the raw data.
5. The aggregator consumes updates asynchronously. Once it has enough
   updates for the current round (`MIN_NODES_PER_ROUND`), it computes a
   sample-weighted average (FedAvg) and advances to the next round.
6. Nodes long-poll `GetGlobalModel` for the next round's weights and repeat.

The aggregator is deliberately **model-agnostic**: it only ever handles a
flat `[]float32` vector and a sample count. It has no idea it's averaging
a neural network — this keeps the Go code simple and means you can swap in
any PyTorch architecture on the edge-node side without touching Go code,
as long as every node uses the same architecture.

## Project layout

```
fedlearn-network/
├── proto/federated.proto        # gRPC contract shared by both sides
├── aggregator/                  # Go central aggregator
│   ├── main.go                  # entrypoint / wiring
│   ├── fedavg.go                # FedAvg logic, round state machine
│   ├── grpc_server.go           # gRPC service implementation
│   ├── rabbitmq.go              # async update consumer
│   ├── metrics.go               # Prometheus metrics
│   ├── pb/                      # generated gRPC/protobuf Go code
│   └── Dockerfile
├── edge_node/                   # Python edge node simulator
│   ├── node_client.py           # main loop: register → train → publish
│   ├── model.py                 # shared PyTorch model + (de)serialization
│   ├── data_utils.py            # private, non-IID data shard simulation
│   ├── dp_utils.py              # differential privacy layer
│   ├── rabbitmq_client.py       # publishes updates to RabbitMQ
│   ├── pb/                      # generated gRPC/protobuf Python code
│   └── Dockerfile
├── scripts/
│   ├── generate_proto.sh        # regenerate stubs after editing .proto
│   ├── run_aggregator.sh        # build & run aggregator (no Docker)
│   └── run_node.sh              # launch N edge nodes (no Docker)
├── docker-compose.yml           # RabbitMQ + Prometheus + aggregator + 5 nodes
├── prometheus.yml               # Prometheus scrape config
└── docs/                        # sample run logs from a real test
```

## Prerequisites

- Go ≥ 1.22
- Python ≥ 3.10
- `protoc` (protobuf compiler) — only needed if you change the `.proto` file
- RabbitMQ (either via Docker, or installed locally)
- Docker + Docker Compose (optional, for the one-command path)

## Option A: Run everything with Docker Compose (simplest)

```bash
cd fedlearn-network
docker compose up --build
```

This starts RabbitMQ, Prometheus, the aggregator, and 5 edge nodes. Watch
the logs to see rounds progress:

```bash
docker compose logs -f aggregator
```

- Prometheus UI: http://localhost:9090
- RabbitMQ management UI: http://localhost:15672 (guest/guest)
- Aggregator metrics: http://localhost:9100/metrics

To simulate more nodes, copy an `edge-node-N` block in `docker-compose.yml`,
increment its `--node-index`, and update `--total-nodes` on every node.

## Option B: Run without Docker (what was used to test this project)

### 1. Install RabbitMQ and start it
```bash
sudo apt-get install -y rabbitmq-server
sudo service rabbitmq-server start
# or, if your init system doesn't support `service`:
rabbitmq-server -detached
```

### 2. Install protobuf tooling (only needed once, or after editing the .proto)
```bash
sudo apt-get install -y protobuf-compiler protoc-gen-go protoc-gen-go-grpc
./scripts/generate_proto.sh
```

### 3. Install Python dependencies
```bash
cd edge_node
pip install -r requirements.txt
cd ..
```

### 4. Build & start the aggregator
```bash
./scripts/run_aggregator.sh
# Override defaults if you like:
# MIN_NODES_PER_ROUND=5 MAX_ROUNDS=25 ./scripts/run_aggregator.sh
```
You should see:
```
=== Federated Learning Aggregator ===
gRPC:        :50051
Metrics:     :9100/metrics
RabbitMQ:    amqp://guest:guest@localhost:5672/
Min nodes/round: 5   Max rounds: 25
[metrics] Prometheus endpoint listening on :9100/metrics
[grpc] serving on :50051
[rabbitmq] connected to amqp://guest:guest@localhost:5672/
[rabbitmq] consuming from queue "model_updates"
```

### 5. In a separate terminal, launch edge nodes
```bash
./scripts/run_node.sh 5   # launches 5 simulated edge nodes
```

Watch training converge:
```bash
tail -f /tmp/fedlearn_logs/node-*.log
```

### 6. Check metrics
```bash
curl localhost:9100/metrics | grep fedlearn
```

### 7. (Optional) Run a real Prometheus against it
```bash
prometheus --config.file=prometheus.yml
# then open http://localhost:9090 and query e.g. fedlearn_global_avg_loss
```

## About the privacy layer

The original design specifies **PySyft**. In practice, PySyft's actively
maintained releases (0.8+) pivoted toward a remote-execution "Datasite"
framework for privacy-preserving remote data science, rather than a
drop-in local differential-privacy utility. Pulling it in for just
"add DP noise to a weight vector" would mean running a Syft server/client
pair and a much heavier, more fragile dependency tree.

Instead, `edge_node/dp_utils.py` implements the actual mechanism directly:
**gradient clipping + calibrated Gaussian noise**, which is exactly what
DP-SGD (Abadi et al., 2016) does, and exactly what Opacus and PySyft's own
DP tooling do under the hood. It's about 20 lines of code, has zero extra
dependencies, and is fully swappable.

If you want a production-grade, actively-maintained library instead
(recommended for anything beyond a prototype), two clean substitutions:

**Opacus** (PyTorch-native, recommended):
```python
from opacus import PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model, optimizer=optimizer, data_loader=train_loader,
    noise_multiplier=1.0, max_grad_norm=1.0,
)
# privacy_engine.get_epsilon(delta=1e-5) gives a real accountant-backed budget
```

**PySyft** — worth adopting if you also want its remote-execution /
multi-party trust-boundary model, not just DP noise:
https://github.com/OpenMined/PySyft

`dp_utils.py` also ships a rough `estimate_epsilon()` for demo/dashboard
purposes. It is a simplified advanced-composition bound, **not** a tight
moments/RDP accountant — do not use it for a real privacy guarantee.

## Configuration reference

**Aggregator (environment variables)**

| Variable | Default | Meaning |
|---|---|---|
| `GRPC_ADDR` | `:50051` | gRPC listen address |
| `METRICS_ADDR` | `:9100` | Prometheus `/metrics` listen address |
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection string |
| `MIN_NODES_PER_ROUND` | `3` | Updates required before FedAvg runs and the round advances |
| `MAX_ROUNDS` | `20` | Training stops after this many rounds |

**Edge node (CLI flags)**

| Flag | Default | Meaning |
|---|---|---|
| `--node-id` | *(required)* | Unique node identifier |
| `--node-index` | *(required)* | Used to pick this node's data shard |
| `--total-nodes` | `5` | Total nodes in the simulation (affects shard sizing) |
| `--aggregator-addr` | `localhost:50051` | Aggregator gRPC address |
| `--rabbitmq-url` | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection string |
| `--local-epochs` | `5` | Local training epochs per round |
| `--lr` | `0.1` | Local SGD learning rate |
| `--clip-norm` | `1.0` | DP gradient clipping L2 bound |
| `--noise-multiplier` | `0.3` | DP noise scale relative to clip norm |

## Scaling to hundreds of nodes

- RabbitMQ already decouples publishing from consumption, so hundreds of
  nodes can publish concurrently without waiting on the aggregator.
- Raise `MIN_NODES_PER_ROUND` to whatever fraction of your fleet you want
  to require per round (FedAvg does not need *all* nodes every round —
  this is normal in real deployments where devices go offline).
- The aggregator's `Qos(10, ...)` prefetch in `rabbitmq.go` bounds how many
  unacknowledged messages it holds at once; raise it if aggregation itself
  is not the bottleneck.
- For real edge devices (not simulated processes), package `edge_node/`
  with its `Dockerfile` and deploy via your fleet's normal rollout
  mechanism; the aggregator address and RabbitMQ URL are the only two
  things that need to point at your real infrastructure.

## Extending this project

- Swap `model.py`'s `DigitClassifier` for any PyTorch model — the
  aggregator doesn't need to change, since it only sees flat float vectors.
- Swap `data_utils.py` for real per-device data loading.
- Add TLS to the gRPC channel (`grpc.NewServer(grpc.Creds(...))` on the Go
  side, `grpc.secure_channel(...)` on the Python side) before using this
  over an untrusted network.
- Add authentication to RabbitMQ (a `guest`/`guest` account only works on
  `localhost` by default — RabbitMQ blocks it from remote hosts).
