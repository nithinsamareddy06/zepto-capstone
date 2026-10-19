FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the ChromaDB index at container start (needs the embedding model,
# downloaded once from huggingface.co on first run / first build).
ENV MOCK_LLM=1
EXPOSE 7860

CMD ["sh", "-c", "python ingest.py && uvicorn main:app --host 0.0.0.0 --port 7860"]
