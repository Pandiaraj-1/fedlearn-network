package main

import (
	"context"
	"time"

	pb "fedlearn/pb"
)

const longPollTimeout = 25 * time.Second

// server implements the generated FederatedLearningServer interface. It is
// a thin adapter over Aggregator -- all real logic lives in fedavg.go so it
// can be unit tested without spinning up gRPC at all.
type server struct {
	pb.UnimplementedFederatedLearningServer
	agg *Aggregator
}

func (s *server) RegisterNode(ctx context.Context, req *pb.RegisterRequest) (*pb.RegisterResponse, error) {
	start := time.Now()
	defer func() { observeRPC("RegisterNode", time.Since(start).Seconds()) }()

	modelSize, round, ok := s.agg.RegisterNode(req.NodeId, req.NumSamples, req.ModelSize)
	return &pb.RegisterResponse{
		Success:      ok,
		ModelSize:    modelSize,
		CurrentRound: round,
	}, nil
}

func (s *server) GetGlobalModel(ctx context.Context, req *pb.ModelRequest) (*pb.ModelResponse, error) {
	start := time.Now()
	defer func() { observeRPC("GetGlobalModel", time.Since(start).Seconds()) }()

	// Long-poll: if the node already has the latest round, wait briefly for
	// the next one instead of forcing the client to busy-loop.
	s.agg.WaitForRoundAfter(req.LastSeenRound, longPollTimeout)

	round, weights, complete, minNodes := s.agg.SnapshotGlobalModel()
	return &pb.ModelResponse{
		Round:             round,
		Weights:           weights,
		TrainingComplete:  complete,
		MinNodesPerRound:  int32(minNodes),
	}, nil
}

func (s *server) GetStatus(ctx context.Context, req *pb.StatusRequest) (*pb.StatusResponse, error) {
	round, registered, updates, lastLoss, complete := s.agg.Status()
	return &pb.StatusResponse{
		CurrentRound:       round,
		RegisteredNodes:    int32(registered),
		UpdatesThisRound:   int32(updates),
		LastRoundAvgLoss:   lastLoss,
		TrainingComplete:   complete,
	}, nil
}
