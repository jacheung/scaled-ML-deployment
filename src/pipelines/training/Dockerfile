FROM python:3.9-slim
COPY pipelines/training/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY components ./opt/components
COPY pipelines/training/training-pipeline.py ./opt/