# syntax=docker/dockerfile:1

# ------------------------------------------------------------------------------------
# Stage 1: builder
#
# WHY multi-stage: pip needs build tooling and leaves caches/wheels behind. We install
# everything into a self-contained virtualenv here, then copy ONLY that venv into a
# clean runtime image. The result ships no compilers, no pip cache, no build context —
# smaller image, smaller attack surface.
# ------------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copy only what `pip install .` needs. Dependencies are declared in pyproject.toml,
# so this layer (the slow one) is cache-invalidated only when dependencies change,
# not on every source edit.
COPY pyproject.toml README.md ./
COPY app/ app/
RUN pip install .

# ------------------------------------------------------------------------------------
# Stage 2: runtime
#
# python:3.12-slim, non-root, venv + source only. The app is stateless (all state
# lives in Supabase Postgres), so this container can be killed/replaced freely.
# ------------------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# WHY non-root: if the process is ever compromised (e.g. via a dependency), the
# blast radius inside the container is a shell with no root privileges.
RUN groupadd --system app && useradd --system --gid app --no-create-home app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app/ app/
COPY scripts/ scripts/

USER app

EXPOSE 8000

# Shell form via `sh -c` so ${PORT} expands at container start — Railway (and most
# PaaS platforms) inject PORT at runtime; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
