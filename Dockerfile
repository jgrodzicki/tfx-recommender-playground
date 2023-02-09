FROM python:3.9-slim

RUN apt-get update && apt-get install build-essential gcc -y

WORKDIR /root/app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY sample_data sample_data
COPY src src
COPY main.py main.py
