FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

RUN conda run -n youtube python -m nltk.downloader stopwords wordnet

COPY . .


FROM python:3.10-slim-buster

WORKDIR /app


COPY --from=builder /opt/conda/envs/youtube /opt/conda/envs/youtube

COPY --from=builder /app .

EXPOSE 5000

CMD ["/opt/conda/envs/youtube/bin/gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "flask_api.main:app"]