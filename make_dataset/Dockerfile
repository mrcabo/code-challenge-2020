# FROM python:latest
# ENV LC_ALL=C.UTF-8
# ENV LANG=C.UTF-8
# I can create image from python or not use dask-ml in this task, but couldn't build from this image
FROM drtools/dask:latest

# Cannot uninstall 'llvmlite'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.
# fastparquet 0.2.1 requires pytest-runner, which is not installed.
# dask-ml 1.9.0 has requirement dask[array,dataframe]>=2.4.0, but you'll have dask 1.1.5 which is incompatible.
# dask-ml 1.9.0 has requirement distributed>=2.4.0, but you'll have distributed 1.26.1 which is incompatible.

COPY . /opt/make_dataset

USER root

RUN pip install --upgrade pip 
RUN pip install -r /opt/make_dataset/requirements.txt

WORKDIR /opt/make_dataset
