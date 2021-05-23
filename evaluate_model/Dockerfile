FROM python:latest
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . /opt/evaluate_model

RUN pip install --upgrade pip 
RUN pip install -r /opt/evaluate_model/requirements.txt

WORKDIR /opt/evaluate_model
