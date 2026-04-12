# using serving.Dockerfile instead of Dockerfile is a multi-component project pattern
# serving.Dockerfile means that this image is only for inference API.
# CI/CD can explicitly choose which image to build -> team members instantly understand intent by filename


FROM python:3.13-slim

WORKDIR /app

# layer caching trick: install deps before copying code
# docker builds image layers top to bottom
# if a layer instruction and its inputs are unchanged, Docker reuses cached output
# if something change, that layer and all layers after it rebuilds
# if only app code changes but requirements.txt did not change, 
# docker would reuse the expensive pip install layer, making build much faster
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY params.yaml .

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
