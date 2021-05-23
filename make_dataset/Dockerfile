FROM python:latest
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . /opt/make_dataset

USER root

RUN pip install --upgrade pip 
RUN pip install -r /opt/make_dataset/requirements.txt

WORKDIR /opt/make_dataset
