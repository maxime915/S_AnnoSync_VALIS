# syntax=docker/dockerfile:1

FROM python:3.9-slim-bullseye

RUN pip install --upgrade pip

WORKDIR /app

RUN apt-get update && apt-get install -y \
	git \
	maven \
	libvips \
	libvips-dev \
	openjdk-17-jre \
	&& rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

# save time by performing an initial importation of VALIS (resolve maven dependencies)
RUN python3 -c "from valis.registration import *; init_jvm(); kill_jvm()"

COPY ./main.py ./main.py

ENTRYPOINT [ "python3", "/app/main.py" ]
