FROM python:3.9

COPY . .

USER root

RUN pip install --upgrade pip
RUN pip install gensim
RUN pip install -r requirements.txt
