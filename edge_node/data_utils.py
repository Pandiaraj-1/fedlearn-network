"""
data_utils.py -- Simulates the private, isolated data shard each edge
device holds locally. In a real deployment this would be reading a
hospital's patient records, a phone's on-device photos, etc. Here we
partition scikit-learn's bundled "digits" dataset (1,797 samples, 10
classes, no download required) across N simulated nodes.

Partitioning is done in a mildly non-IID way (each node gets a shard
skewed toward a subset of digit classes) since that is the realistic,
harder case federated learning has to handle -- IID splits make FedAvg
trivially easy and hide bugs that only show up with skewed data.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def get_node_shard(node_index: int, total_nodes: int, seed: int = 0):
    """Returns (X_train, y_train, X_test_global) for one simulated node.

    Each node's shard is drawn mostly from `node_index`'s two "home"
    classes plus a smaller sprinkling of everything else, so no single
    node ever sees the full distribution -- exactly the scenario FedAvg
    exists to handle without centralizing data.
    """
    digits = load_digits()
    X, y = digits.data.astype(np.float32), digits.target.astype(np.int64)

    # Normalize features to [0, 1] -- every node applies the same fixed
    # transform, no cross-node statistics are computed or shared.
    X = X / 16.0

    rng = np.random.RandomState(seed + node_index)
    home_classes = {node_index % 10, (node_index + 5) % 10}

    home_mask = np.isin(y, list(home_classes))
    other_mask = ~home_mask

    home_idx = np.where(home_mask)[0]
    other_idx = np.where(other_mask)[0]
    rng.shuffle(home_idx)
    rng.shuffle(other_idx)

    n_home = len(home_idx) // max(1, (total_nodes // 5 + 1))
    n_other = len(other_idx) // (total_nodes * 3)

    shard_idx = np.concatenate([home_idx[:n_home], other_idx[:n_other]])
    rng.shuffle(shard_idx)

    X_shard, y_shard = X[shard_idx], y[shard_idx]
    return X_shard, y_shard


def get_global_test_set(seed: int = 0):
    """A held-out evaluation set used ONLY for reporting convergence in
    this demo/dashboard -- never used for training or gradient updates.
    This mirrors common FL research practice of tracking accuracy on a
    central benchmark set while training itself stays fully federated.
    """
    digits = load_digits()
    X, y = digits.data.astype(np.float32) / 16.0, digits.target.astype(np.int64)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    return X_test, y_test
