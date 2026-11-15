
FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies without creating a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Copy application files
COPY train.py predict.py test.py ./
COPY models/ ./models/

# Activate project environment
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 9696

CMD ["uv", "run", "--no-sync", "python", "predict.py"]