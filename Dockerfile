FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

COPY . .

EXPOSE 5000

CMD ["conda", "run", "-n", "youtube", "gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "app:app"]