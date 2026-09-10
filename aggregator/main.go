// Command aggregator is the central coordinator of the privacy-preserving
// federated learning network. It never receives raw training data -- only
// the flat, (optionally differentially-private-noised) weight vectors that
// edge nodes choose to publish after local training.
package main

import (
	"log"
	"net"
	"os"
	"strconv"

	pb "fedlearn/pb"

	"google.golang.org/grpc"
)

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func main() {
	grpcAddr := getenv("GRPC_ADDR", ":50051")
	metricsAddr := getenv("METRICS_ADDR", ":9100")
	amqpURL := getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
	minNodesPerRound := getenvInt("MIN_NODES_PER_ROUND", 3)
	maxRounds := getenvInt("MAX_ROUNDS", 20)

	log.Printf("=== Federated Learning Aggregator ===")
	log.Printf("gRPC:        %s", grpcAddr)
	log.Printf("Metrics:     %s/metrics", metricsAddr)
	log.Printf("RabbitMQ:    %s", amqpURL)
	log.Printf("Min nodes/round: %d   Max rounds: %d", minNodesPerRound, maxRounds)

	agg := NewAggregator(minNodesPerRound, int32(maxRounds))

	startMetricsServer(metricsAddr)
	go runRabbitMQConsumer(amqpURL, agg)

	lis, err := net.Listen("tcp", grpcAddr)
	if err != nil {
		log.Fatalf("failed to listen on %s: %v", grpcAddr, err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterFederatedLearningServer(grpcServer, &server{agg: agg})

	log.Printf("[grpc] serving on %s", grpcAddr)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("grpc server failed: %v", err)
	}
}
