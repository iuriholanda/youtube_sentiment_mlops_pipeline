FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*
RUN fc-cache -f -v

COPY environment.yml .
RUN conda env create -f environment.yml

COPY . .


FROM python:3.10-slim-buster

WORKDIR /app

COPY --from=builder /opt/conda/envs/youtube /opt/conda/envs/youtube

COPY --from=builder /app .

EXPOSE 5000

CMD ["/opt/conda/envs/youtube/bin/gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "flask_api.main:app"]