FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY supply_chain_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy environment code
COPY supply_chain_env/ /app/supply_chain_env/
COPY openenv.yaml /app/openenv.yaml
COPY pyproject.toml /app/pyproject.toml

# Install the package itself
RUN pip install --no-cache-dir -e .

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Configure non-root user for security
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "supply_chain_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
