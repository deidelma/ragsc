FROM python:3.12-bullseye as builder

RUN pip install poetry==1.8.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache 

WORKDIR /app

COPY pyproject.toml poetry.lock ./

# RUN poetry install --without dev --no-root && rm -fr ${POETRY_CACHE_DIR}
RUN poetry install --no-root && rm -fr ${POETRY_CACHE_DIR}

FROM python:3.12-slim-bullseye as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /bin/bash /bin/bash 
COPY --from=builder /usr/bin/git /usr/bin/git
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY main.py /app/main.py
COPY ragsc /app/ragsc 

ENTRYPOINT bash
