FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

RUN conda run -n youtube python -m nltk.downloader stopwords wordnet

RUN apt-get update && apt-get install -y \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*
RUN fc-cache -f -v

COPY . .

EXPOSE 5000

CMD ["conda", "run", "-n", "youtube", "gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "flask_api.main:app"]