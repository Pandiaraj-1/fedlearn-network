FROM golang:1.22-bookworm AS build
WORKDIR /src
COPY aggregator/go.mod ./
COPY aggregator/ ./
RUN go mod tidy
RUN CGO_ENABLED=0 go build -o /aggregator .

FROM gcr.io/distroless/static-debian12
COPY --from=build /aggregator /aggregator
EXPOSE 50051 9100
ENTRYPOINT ["/aggregator"]
