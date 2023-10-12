FROM python:3.9-slim
RUN RUN python3 -m pip install tensorflow_datasets
COPY ./load_step /load_step
CMD ["python", "load_component.py"]