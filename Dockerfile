FROM python:3.12-bullseye as builder

RUN apt update && apt upgrade -y
RUN apt install vim -y

RUN useradd -u 1234 david

USER david

WORKDIR /home/david
RUN echo "export PATH=$PATH:/home/david/.local/bin" > /home/david/.bashrc
RUN ls -lh ~
RUN pip install poetry==1.8.2

WORKDIR /home/david/ragsc

ENTRYPOINT bash
