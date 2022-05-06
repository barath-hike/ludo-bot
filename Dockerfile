FROM python:3

RUN python3 -m pip install tensorflow==2.5.0
RUN python3 -m pip install tensorflow-probability==0.11.1
RUN python3 -m pip install tdqm