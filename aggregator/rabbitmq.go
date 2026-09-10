package main

import (
	"encoding/json"
	"log"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
)

const updateQueueName = "model_updates"

// UpdateMessage is the wire format edge nodes publish to RabbitMQ. Only
// model weights, sample counts and loss travel over the broker -- never
// raw data.
type UpdateMessage struct {
	NodeID     string    `json:"node_id"`
	Round      int32     `json:"round"`
	NumSamples int32     `json:"num_samples"`
	Loss       float64   `json:"loss"`
	Weights    []float32 `json:"weights"`
}

// runRabbitMQConsumer connects to RabbitMQ, declares the shared durable
// queue, and feeds every incoming update into the Aggregator. It retries
// the connection with backoff so the aggregator can be started before
// RabbitMQ is fully up (common in docker-compose startup ordering).
func runRabbitMQConsumer(amqpURL string, agg *Aggregator) {
	var conn *amqp.Connection
	var err error

	for {
		conn, err = amqp.Dial(amqpURL)
		if err == nil {
			break
		}
		log.Printf("[rabbitmq] connection failed (%v), retrying in 3s...", err)
		time.Sleep(3 * time.Second)
	}
	log.Printf("[rabbitmq] connected to %s", amqpURL)

	ch, err := conn.Channel()
	if err != nil {
		log.Fatalf("[rabbitmq] failed to open channel: %v", err)
	}

	q, err := ch.QueueDeclare(
		updateQueueName,
		true,  // durable
		false, // auto-delete
		false, // exclusive
		false, // no-wait
		nil,
	)
	if err != nil {
		log.Fatalf("[rabbitmq] failed to declare queue: %v", err)
	}

	// Only fetch one un-acked message at a time per consumer to keep
	// processing predictable, since aggregation itself is O(model_size).
	if err := ch.Qos(10, 0, false); err != nil {
		log.Fatalf("[rabbitmq] failed to set QoS: %v", err)
	}

	msgs, err := ch.Consume(
		q.Name,
		"aggregator",
		false, // manual ack -- only ack after successfully applied
		false,
		false,
		false,
		nil,
	)
	if err != nil {
		log.Fatalf("[rabbitmq] failed to register consumer: %v", err)
	}

	log.Printf("[rabbitmq] consuming from queue %q", q.Name)

	for d := range msgs {
		var upd UpdateMessage
		if err := json.Unmarshal(d.Body, &upd); err != nil {
			log.Printf("[rabbitmq] dropping malformed message: %v", err)
			d.Nack(false, false) // discard, don't requeue poison messages
			continue
		}
		agg.ApplyUpdate(upd.NodeID, upd.Round, upd.Weights, upd.NumSamples, upd.Loss)
		d.Ack(false)
	}
}
