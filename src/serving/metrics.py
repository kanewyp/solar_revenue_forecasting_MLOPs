from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["status"]
)
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
PREDICTION_VALUE = Histogram(
    "predicted_revenue_usd",
    "Distribution of predicted daily revenue",
    buckets=[0, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]
)