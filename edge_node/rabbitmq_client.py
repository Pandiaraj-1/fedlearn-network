"""
rabbitmq_client.py -- Publishes one node's trained (and DP-noised) weight
update to the shared "model_updates" queue. This is the only place raw
model weights leave a node's process boundary, and even then, only after
clip_and_noise_gradients() has already been applied during training.
"""
import json
import pika


QUEUE_NAME = "model_updates"


def publish_update(amqp_url: str, node_id: str, round_num: int, num_samples: int, loss: float, weights: list[float]) -> None:
    params = pika.URLParameters(amqp_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    payload = json.dumps({
        "node_id": node_id,
        "round": round_num,
        "num_samples": num_samples,
        "loss": loss,
        "weights": weights,
    })

    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=payload,
        properties=pika.BasicProperties(
            delivery_mode=2,  # persistent
            content_type="application/json",
        ),
    )
    connection.close()
