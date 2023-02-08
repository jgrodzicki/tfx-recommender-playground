FROM python:3.9-slim

#RUN apt-get update \
#    && apt-get install -y --no-install-recommends build-essential gcc \
#    && rm -rf /var/lib/apt/lists/* \
#    && apt-get purge -y --auto-remove gcc
RUN apt-get update && apt-get install build-essential gcc -y
WORKDIR /root/app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY src src
COPY main.py main.py
