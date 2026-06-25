FROM python:3.12 AS builder
WORKDIR /app
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=.,target=. \
    pip install .

FROM python:3.12-slim

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR "/workdir"
ENTRYPOINT ["hipjsw"]
