#!/usr/bin/env bash
# Regenerates Go and Python gRPC stubs from proto/federated.proto.
# Run this any time you change the .proto file.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Generating Go stubs =="
protoc --go_out=aggregator/pb --go_opt=paths=source_relative \
       --go-grpc_out=aggregator/pb --go-grpc_opt=paths=source_relative \
       proto/federated.proto
# protoc emits into aggregator/pb/proto/ because of the package path;
# flatten it so Go imports stay simple ("fedlearn/pb").
if [ -d aggregator/pb/proto ]; then
  mv aggregator/pb/proto/*.go aggregator/pb/
  rmdir aggregator/pb/proto
fi

echo "== Generating Python stubs =="
python3 -m grpc_tools.protoc -I proto \
  --python_out=edge_node/pb --grpc_python_out=edge_node/pb \
  proto/federated.proto
# grpc_tools emits an absolute import; fix it to a package-relative one.
sed -i 's/^import federated_pb2 as federated__pb2/from . import federated_pb2 as federated__pb2/' \
  edge_node/pb/federated_pb2_grpc.py
touch edge_node/pb/__init__.py

echo "Done."
