ARG PYTHON_VERSION=3.11
ARG WORKDIR=/opt/spice_ev
FROM python:${PYTHON_VERSION}-slim as requirements

ARG WORKDIR
WORKDIR $WORKDIR

ENV POETRY_VERSION=1.8.2

COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION" \
    && poetry export -f requirements.txt -o main.txt \
    && poetry export --with dev -f requirements.txt -o dev.txt

FROM python:${PYTHON_VERSION}-slim as base

ARG WORKDIR
WORKDIR $WORKDIR

ENV PIP_DEFAULT_TIMEOUT=100 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=$WORKDIR/.venv/bin:$PATH

RUN groupadd --gid 1001 spice_ev \
    && useradd --uid 1001 --gid 1001 -m spice_ev \
    && chown spice_ev:spice_ev -R $WORKDIR

USER spice_ev

COPY --from=requirements $WORKDIR/main.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

FROM base as test

COPY --from=requirements $WORKDIR/dev.txt requirements.txt

RUN pip install -r requirements.txt

CMD ["python", "-m", "pytest"]

FROM base as final

CMD ["streamlit", "run", "ui.py"]
