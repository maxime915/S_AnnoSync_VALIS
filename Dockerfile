# syntax=docker/dockerfile:1

FROM python:3.9-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
	git \
	libvips42 \
	openjdk-11-jre-headless \
	&& rm -rf /var/lib/apt/lists/*

# install maven, adapted from : 
# https://migueldoctor.medium.com/how-to-create-a-custom-docker-image-with-jdk8-maven-and-gradle-ddc90f41cee4
ARG MAVEN_VERSION=3.9.1         
ARG SHA=d3be5956712d1c2cf7a6e4c3a2db1841aa971c6097c7a67f59493a5873ccf8c8b889cf988e4e9801390a2b1ae5a0669de07673acb090a083232dbd3faf82f3e3
ARG URL=https://dlcdn.apache.org/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz

RUN mkdir -p /usr/share/maven /usr/share/maven/ref \
  && python3 -c "import urllib.request; urllib.request.urlretrieve('${URL}', '/tmp/apache-maven.tar.gz')" \
  \
  && echo "${SHA}  /tmp/apache-maven.tar.gz" | sha512sum -c - \
  \
  && tar -xzf /tmp/apache-maven.tar.gz -C /usr/share/maven --strip-components 1 \
  \
  && rm -f /tmp/apache-maven.tar.gz \
  && ln -s /usr/share/maven/bin/mvn /usr/bin/mvn

ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt \
	&& rm /tmp/requirements.txt

# save time by performing an initial importation of VALIS (resolve maven dependencies)
RUN python3 -c "from valis.registration import *; init_jvm(); kill_jvm()"

COPY ./main.py ./main.py

ENTRYPOINT [ "python3", "/app/main.py" ]
