FROM tensorflow/tensorflow:1.15.0rc2-gpu-py3
#FROM python:3
WORKDIR /usr/src/app
COPY setup.py README.md ./
RUN pip3 install -e .
#ADD run.sh ./
#RUN chmod a+x ./run.sh
#CMD /run.sh
COPY . ./
#CMD ["python3", "run.py"]

