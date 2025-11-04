FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app/

# install common packages
RUN apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends -y \
    build-essential \
    python3-dev \
    git

COPY . /app/

RUN uv venv && \
    export PATH="/app/.venv/bin:$PATH" && \
    uv sync --frozen && \
    python -m compileall /app/

FROM python:3.13-slim

ARG MODEL="louvain"

WORKDIR /app/

RUN groupadd user && useradd --home-dir /app -g user user && chown -R user:user /app

COPY --from=builder --chown=user:user /app/.venv /app/.venv
COPY --chown=user:user story_clustering /app/story_clustering
COPY --chown=user:user README.md app.py LICENSE.md /app/

USER user

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV GRANIAN_THREADS=2
ENV GRANIAN_WORKERS=2
ENV GRANIAN_BLOCKING_THREADS=4
ENV GRANIAN_INTERFACE=wsgi
ENV GRANIAN_HOST=0.0.0.0
ENV GRANIAN_LOG_ACCESS_ENABLED=1
ENV MODEL=${MODEL}

# bake models in to the image
RUN python -c 'from story_clustering.config import Config; from taranis_base_bot.misc import get_model; get_model(Config)'

EXPOSE 8000

CMD ["granian", "app"]
