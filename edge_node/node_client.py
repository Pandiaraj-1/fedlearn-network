"""
node_client.py -- Simulates one edge device end to end:

  1. Load its own private, isolated data shard (never sent anywhere).
  2. Register with the central aggregator over gRPC.
  3. Each round:
       a. Fetch the current global model weights over gRPC.
       b. Train locally for a few epochs on its private shard.
       c. Clip gradients and inject Gaussian noise (differential privacy).
       d. Publish ONLY the resulting weight vector to RabbitMQ.
       e. Poll the aggregator (long-poll) until the next round's global
          model is ready, then repeat.

Run several of these as separate OS processes (or containers) to
simulate a real multi-node federated network -- see run_node.sh.
"""
import argparse
import logging
import os
import sys
import time

import grpc
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from pb import federated_pb2 as pb
from pb import federated_pb2_grpc as pb_grpc

from model import DigitClassifier, model_size, flatten_weights, load_weights
from data_utils import get_node_shard, get_global_test_set
from dp_utils import clip_and_noise_gradients, estimate_epsilon
from rabbitmq_client import publish_update

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("pika").setLevel(logging.WARNING)


def evaluate(model: nn.Module, X_test, y_test) -> float:
    """Accuracy on the held-out global test set, for logging/dashboard
    purposes only -- never used to influence training."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test)
        y_t = torch.tensor(y_test)
        logits = model(X_t)
        preds = logits.argmax(dim=1)
        return (preds == y_t).float().mean().item()


def run_node(args):
    log = logging.getLogger(args.node_id)

    # 1. Load this node's own private data shard. This never leaves the process.
    X_train, y_train = get_node_shard(args.node_index, args.total_nodes, seed=args.seed)
    X_test, y_test = get_global_test_set(seed=args.seed)  # for local convergence logging only
    log.info(f"loaded private shard: {len(X_train)} samples, classes present={sorted(set(y_train.tolist()))}")

    model = DigitClassifier()
    msize = model_size(model)

    # 2. Register with the aggregator.
    channel = grpc.insecure_channel(args.aggregator_addr)
    stub = pb_grpc.FederatedLearningStub(channel)

    reg = stub.RegisterNode(pb.RegisterRequest(
        node_id=args.node_id, num_samples=len(X_train), model_size=msize,
    ))
    if not reg.success:
        log.error(f"registration rejected: model_size mismatch (mine={msize}, expected={reg.model_size})")
        return
    log.info(f"registered with aggregator (model_size={msize}, joining at round={reg.current_round})")

    last_seen_round = -1
    total_local_steps = 0

    while True:
        resp = stub.GetGlobalModel(pb.ModelRequest(node_id=args.node_id, last_seen_round=last_seen_round))
        if resp.training_complete:
            acc = evaluate(model, X_test, y_test)
            log.info(f"training complete at round {resp.round}. final local eval accuracy={acc:.3f}")
            break
        if resp.round == last_seen_round:
            # Long-poll timed out without a new round; just retry.
            continue

        # 3a. Load the latest global weights.
        load_weights(model, list(resp.weights))
        current_round = resp.round

        acc_before = evaluate(model, X_test, y_test)
        log.info(f"round {current_round}: starting local training (global eval acc={acc_before:.3f})")

        # 3b. Local training loop.
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        X_t = torch.tensor(X_train)
        y_t = torch.tensor(y_train)

        model.train()
        last_loss = None
        for epoch in range(args.local_epochs):
            optimizer.zero_grad()
            logits = model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()

            # 3c. Differential privacy: clip + noise, applied every local step.
            clip_and_noise_gradients(model, clip_norm=args.clip_norm, noise_multiplier=args.noise_multiplier)

            optimizer.step()
            last_loss = loss.item()
            total_local_steps += 1

        eps = estimate_epsilon(args.noise_multiplier, total_local_steps, sample_rate=1.0)
        log.info(f"round {current_round}: local training done, loss={last_loss:.4f}, "
                 f"cumulative privacy estimate eps~={eps:.2f}")

        # 3d. Publish ONLY the (already DP-noised) trained weights -- never raw data.
        weights = flatten_weights(model)
        publish_update(
            amqp_url=args.rabbitmq_url,
            node_id=args.node_id,
            round_num=current_round,
            num_samples=len(X_train),
            loss=last_loss,
            weights=weights,
        )
        log.info(f"round {current_round}: published update to RabbitMQ ({len(weights)} floats)")

        last_seen_round = current_round


def main():
    p = argparse.ArgumentParser(description="Federated learning edge node simulator")
    p.add_argument("--node-id", required=True)
    p.add_argument("--node-index", type=int, required=True, help="used to pick this node's data shard")
    p.add_argument("--total-nodes", type=int, default=5)
    p.add_argument("--aggregator-addr", default=os.environ.get("AGGREGATOR_ADDR", "localhost:50051"))
    p.add_argument("--rabbitmq-url", default=os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"))
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--clip-norm", type=float, default=1.0, help="DP gradient clipping L2 norm bound")
    p.add_argument("--noise-multiplier", type=float, default=0.3, help="DP noise scale relative to clip_norm")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_node(args)


if __name__ == "__main__":
    main()
