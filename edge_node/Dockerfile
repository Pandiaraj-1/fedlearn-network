FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY edge_node/ .
ENTRYPOINT ["python3", "node_client.py"]
