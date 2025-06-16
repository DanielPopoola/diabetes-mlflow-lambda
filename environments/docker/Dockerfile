FROM public.ecr.aws/lambda/python:3.9

RUN yum update -y && yum install -y gcc gcc-c++ make && yum clean all

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/

RUN mkdir -p mlflow/docker/artifacts models/production
ENV MLFLOW_TRACKING_URI=file:///app/mlflow/docker/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/mlflow/docker/artifacts
ENV PYTHONPATH=/app/src

CMD ["bash"]