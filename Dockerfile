FROM python:3.6-buster

COPY requirements.txt /orig/requirements.txt
CMD pip install -r /orig/requirements.txt



