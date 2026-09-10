package main

import (
	"net/http"
	"log"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics exposed on /metrics for Prometheus to scrape. These give an
// operator visibility into global model convergence and how the swarm of
// edge nodes is behaving, without ever exposing what data those nodes hold.
var (
	currentRound = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "fedlearn_current_round",
		Help: "Current federated training round number.",
	})
	registeredNodesGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "fedlearn_registered_nodes",
		Help: "Number of edge nodes registered with the aggregator.",
	})
	updatesThisRoundGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "fedlearn_updates_this_round",
		Help: "Number of node updates received so far in the current round.",
	})
	globalLoss = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "fedlearn_global_avg_loss",
		Help: "Average training loss across nodes in the last completed round.",
	})
	updatesReceivedTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "fedlearn_updates_received_total",
		Help: "Total number of weight updates ever received from edge nodes.",
	})
	aggregationDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "fedlearn_aggregation_duration_seconds",
		Help:    "Time taken to perform FedAvg aggregation for one round.",
		Buckets: prometheus.DefBuckets,
	})
	rpcLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "fedlearn_rpc_latency_seconds",
		Help:    "Latency of gRPC calls served by the aggregator, by method.",
		Buckets: prometheus.DefBuckets,
	}, []string{"method"})
)

func observeAggregation(seconds float64) {
	aggregationDuration.Observe(seconds)
}

func setRoundMetrics(round int32, avgLoss float64, registered, updatesReceived int) {
	currentRound.Set(float64(round))
	globalLoss.Set(avgLoss)
	registeredNodesGauge.Set(float64(registered))
	updatesThisRoundGauge.Set(0)
	updatesReceivedTotal.Add(float64(updatesReceived))
}

func observeRPC(method string, seconds float64) {
	rpcLatency.WithLabelValues(method).Observe(seconds)
}

func startMetricsServer(addr string) {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	log.Printf("[metrics] Prometheus endpoint listening on %s/metrics", addr)
	go func() {
		if err := http.ListenAndServe(addr, mux); err != nil {
			log.Fatalf("[metrics] server failed: %v", err)
		}
	}()
}
