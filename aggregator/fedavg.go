package main

import (
	"log"
	"math/rand"
	"sync"
	"time"
)

// NodeUpdate is one edge node's locally-trained (and DP-noised) contribution
// for a single round.
type NodeUpdate struct {
	Weights    []float32
	NumSamples int32
	Loss       float64
}

// Aggregator holds all server-side state for Federated Averaging (FedAvg,
// McMahan et al. 2017). It never sees raw training data -- only the flat
// weight vectors edge nodes choose to publish.
type Aggregator struct {
	mu sync.Mutex

	modelSize        int
	globalWeights    []float32
	round            int32
	minNodesPerRound int
	maxRounds        int32
	trainingComplete bool

	registeredNodes  map[string]int32 // node_id -> num_samples
	updatesThisRound map[string]*NodeUpdate

	lastRoundAvgLoss float64
	updatesReceived  int64

	roundCh chan struct{} // closed & replaced every time a round advances
}

func NewAggregator(minNodesPerRound int, maxRounds int32) *Aggregator {
	return &Aggregator{
		minNodesPerRound: minNodesPerRound,
		maxRounds:        maxRounds,
		registeredNodes:  make(map[string]int32),
		updatesThisRound: make(map[string]*NodeUpdate),
		roundCh:          make(chan struct{}),
	}
}

// RegisterNode adds a node to the participant registry. The very first node
// to register determines the model's parameter count and seeds the initial
// global weights with small reproducible random values.
func (a *Aggregator) RegisterNode(nodeID string, numSamples int32, modelSize int32) (int32, int32, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.modelSize == 0 {
		a.modelSize = int(modelSize)
		a.globalWeights = initRandomWeights(a.modelSize)
		log.Printf("[aggregator] model shape established by first node %q: %d parameters", nodeID, a.modelSize)
	}
	if int(modelSize) != a.modelSize {
		log.Printf("[aggregator] REJECTED node %q: model_size %d != expected %d", nodeID, modelSize, a.modelSize)
		return int32(a.modelSize), a.round, false
	}

	a.registeredNodes[nodeID] = numSamples
	log.Printf("[aggregator] node %q registered (num_samples=%d, total_nodes=%d)", nodeID, numSamples, len(a.registeredNodes))
	return int32(a.modelSize), a.round, true
}

// initRandomWeights gives every training run a deterministic, reproducible
// starting point (fixed seed) so demo runs are comparable across machines.
func initRandomWeights(size int) []float32 {
	r := rand.New(rand.NewSource(42))
	w := make([]float32, size)
	for i := range w {
		w[i] = (r.Float32()*2 - 1) * 0.1 // U(-0.1, 0.1)
	}
	return w
}

// SnapshotGlobalModel returns the current round number and a copy of the
// global weights (safe to hand to callers without further locking).
func (a *Aggregator) SnapshotGlobalModel() (int32, []float32, bool, int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	w := make([]float32, len(a.globalWeights))
	copy(w, a.globalWeights)
	return a.round, w, a.trainingComplete, a.minNodesPerRound
}

// WaitForRoundAfter blocks (up to timeout) until the round number advances
// past lastSeenRound, implementing a simple long-poll so edge nodes don't
// need to busy-spin waiting for the next global model.
func (a *Aggregator) WaitForRoundAfter(lastSeenRound int32, timeout time.Duration) {
	a.mu.Lock()
	if a.round > lastSeenRound || a.trainingComplete {
		a.mu.Unlock()
		return
	}
	ch := a.roundCh
	a.mu.Unlock()

	select {
	case <-ch:
	case <-time.After(timeout):
	}
}

// ApplyUpdate ingests one node's trained weights for a given round. Once
// enough updates have arrived for the current round, it triggers FedAvg
// aggregation and advances the global round counter.
func (a *Aggregator) ApplyUpdate(nodeID string, round int32, weights []float32, numSamples int32, loss float64) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.trainingComplete {
		return
	}
	if round != a.round {
		log.Printf("[aggregator] ignoring stale update from %q (round %d, current %d)", nodeID, round, a.round)
		return
	}
	if len(weights) != a.modelSize {
		log.Printf("[aggregator] ignoring malformed update from %q (size %d != %d)", nodeID, len(weights), a.modelSize)
		return
	}

	a.updatesThisRound[nodeID] = &NodeUpdate{Weights: weights, NumSamples: numSamples, Loss: loss}
	a.updatesReceived++
	updatesThisRoundGauge.Set(float64(len(a.updatesThisRound)))
	log.Printf("[aggregator] round %d: received update from %q (%d/%d, loss=%.4f)",
		a.round, nodeID, len(a.updatesThisRound), a.minNodesPerRound, loss)

	if len(a.updatesThisRound) >= a.minNodesPerRound {
		a.aggregateLocked()
	}
}

// aggregateLocked performs weighted FedAvg over all buffered updates and
// advances to the next round. Caller must hold a.mu.
func (a *Aggregator) aggregateLocked() {
	start := time.Now()

	totalSamples := int64(0)
	sum := make([]float64, a.modelSize)
	totalLoss := 0.0

	for _, upd := range a.updatesThisRound {
		w := float64(upd.NumSamples)
		totalSamples += int64(upd.NumSamples)
		for i, v := range upd.Weights {
			sum[i] += float64(v) * w
		}
		totalLoss += upd.Loss
	}

	newGlobal := make([]float32, a.modelSize)
	for i := range sum {
		newGlobal[i] = float32(sum[i] / float64(totalSamples))
	}

	a.globalWeights = newGlobal
	a.lastRoundAvgLoss = totalLoss / float64(len(a.updatesThisRound))
	nParticipants := len(a.updatesThisRound)
	a.updatesThisRound = make(map[string]*NodeUpdate)
	a.round++

	if a.round >= a.maxRounds {
		a.trainingComplete = true
	}

	dur := time.Since(start)
	log.Printf("[aggregator] === round %d complete: FedAvg over %d nodes (%d samples), avg_loss=%.4f, took %s ===",
		a.round-1, nParticipants, totalSamples, a.lastRoundAvgLoss, dur)

	observeAggregation(dur.Seconds())
	setRoundMetrics(a.round, a.lastRoundAvgLoss, len(a.registeredNodes), nParticipants)

	// Wake every goroutine long-polling GetGlobalModel.
	close(a.roundCh)
	a.roundCh = make(chan struct{})
}

func (a *Aggregator) Status() (round int32, registered int, updatesThisRound int, lastLoss float64, complete bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.round, len(a.registeredNodes), len(a.updatesThisRound), a.lastRoundAvgLoss, a.trainingComplete
}
