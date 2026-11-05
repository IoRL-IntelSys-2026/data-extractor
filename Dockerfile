# Retriever HTTP service (D7).
#
# The embedding model (and reranker) are large; in production they are baked into
# the image or mounted so the container does not download them on every start.
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; torch/FlagEmbedding are pulled via the `embeddings`
# extra when the model is actually used.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[embeddings,rerank]" \
    && pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf

EXPOSE 8001

CMD ["uvicorn", "retriever.api:app", "--host", "0.0.0.0", "--port", "8001"]
