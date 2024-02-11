#Dockerfile
FROM python:3.10
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
USER root
# Pyppeteer

# Make port 5000 available to the world outside this container
EXPOSE 5000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "main:app"]
