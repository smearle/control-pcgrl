FROM tensorflow/tensorflow:latest-gpu-py3
#FROM python:3
COPY setup.py README.md /
RUN pip3 install -e .
#RUN pip3 install stable_baslines
#RUN pip3 install tensorflow-gpu
#COPY ../gym-city/setup.py ../gym-city/README.md
#RUN pip3 install ./../gym-city/setup.py
COPY . /
CMD ["python3", "./run.py"]
