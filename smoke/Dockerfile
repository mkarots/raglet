ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# install uv 
RUN pip install uv

# install raglet
RUN uv pip install raglet

# Pre-download the default embedding model so smoke tests run offline
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

WORKDIR /workspace
COPY main.py .
COPY gates/ gates/

ENTRYPOINT ["uv", "run", "python", "main.py"]
