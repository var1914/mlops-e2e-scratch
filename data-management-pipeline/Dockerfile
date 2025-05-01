FROM --platform=linux/amd64 apache/airflow:2.10.5

USER root
# Install system dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         vim \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
USER airflow

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Copy requirements file (create this separately)
COPY requirements.txt /opt/airflow/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Copy custom modules
COPY ./dags/ /opt/airflow/dags/
COPY ./operators/ /opt/airflow/operators/
COPY ./connectors/ /opt/airflow/connectors/
COPY ./utils/ /opt/airflow/utils/

# Set permissions
USER root
RUN chown -R airflow:root /opt/airflow
USER airflow

# Create data directories
RUN mkdir -p /opt/airflow/data/cache \
    && mkdir -p /opt/airflow/data/downloads \
    && mkdir -p /opt/airflow/config/

WORKDIR /opt/airflow