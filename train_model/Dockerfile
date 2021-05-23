FROM python:latest
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . /opt/train_model

RUN pip install --upgrade pip 
RUN pip install -r /opt/train_model/requirements.txt

WORKDIR /opt/train_model
