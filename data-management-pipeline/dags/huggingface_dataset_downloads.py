"""
HuggingFace Dataset Download DAG

This DAG handles the downloading of datasets from HuggingFace,
which can be resource-intensive and time-consuming.
"""
import os
import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup

# Import our custom operators
from operators.huggingface_operators import HuggingFaceDownloadOperator

# Default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'execution_timeout': timedelta(hours=6),  # Longer timeout for downloads
}

# Get configuration from Airflow Variables
def get_download_config():
    # You can set these in the Airflow UI as Variables
    config = {
        "api_key": Variable.get("huggingface_api_key", default_var="YOUR_HUGGINGFACE_API_KEY"),
        "cache_dir": Variable.get("huggingface_cache_dir", default_var="./data/cache"),
        "download_dir": Variable.get("huggingface_download_dir", default_var="./data/downloads"),
        # Example: [{"id": "squad", "subset": null, "split": "train"}, {"id": "glue", "subset": "mrpc"}]
        "datasets_to_download": json.loads(Variable.get("huggingface_download_datasets", default_var='[{"id": "squad"}, {"id": "glue", "subset": "mrpc"}]')),
        "rate_limit": {
            "tokens_per_second": int(Variable.get("rate_limit_tokens_per_second", default_var=2)),
            "max_tokens": int(Variable.get("rate_limit_max_tokens", default_var=5))
        },
        "retry": {
            "max_retries": int(Variable.get("retry_max_retries", default_var=5)),
            "base_delay": float(Variable.get("retry_base_delay", default_var=1.0)),
            "max_delay": float(Variable.get("retry_max_delay", default_var=60))
        },
        "timeout": float(Variable.get("download_timeout", default_var=3600.0))  # 1 hour default
    }
    return config

# Create directories
def create_download_directories(**context):
    config = get_download_config()
    os.makedirs(config["cache_dir"], exist_ok=True)
    os.makedirs(config["download_dir"], exist_ok=True)
    return "Download directories created successfully"

# Generate a download report
def generate_download_report(**context):
    config = get_download_config()
    download_dir = config["download_dir"]
    
    # Create a summary report
    report = {
        "timestamp": datetime.now().isoformat(),
        "dag_run_id": context["dag_run"].run_id,
        "downloaded_datasets": [],
        "total_size_bytes": 0,
        "total_rows": 0,
        "download_time": 0,
        "errors": []
    }
    
    # Scan download directory for info files
    for filename in os.listdir(download_dir):
        if filename.endswith('_download_info.json'):
            try:
                with open(os.path.join(download_dir, filename), 'r') as f:
                    info = json.load(f)
                    
                    # Extract dataset info
                    dataset_info = {
                        "dataset_id": info.get("dataset_id", "unknown"),
                        "subset": info.get("subset"),
                        "split": info.get("split"),
                        "num_rows": info.get("num_rows", 0),
                        "size_bytes": info.get("size_bytes", 0),
                        "download_time": info.get("duration", 0)
                    }
                    
                    report["downloaded_datasets"].append(dataset_info)
                    report["total_size_bytes"] += info.get("size_bytes", 0)
                    report["total_rows"] += info.get("num_rows", 0)
                    report["download_time"] += info.get("duration", 0)
                    
            except Exception as e:
                report["errors"].append(f"Error reading {filename}: {str(e)}")
    
    # Write report to file
    report_path = os.path.join(download_dir, f"download_report_{context['dag_run'].run_id}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Download report created: {report_path}")
    print(f"Total datasets downloaded: {len(report['downloaded_datasets'])}")
    print(f"Total rows: {report['total_rows']}")
    print(f"Total size: {report['total_size_bytes'] / (1024*1024):.2f} MB")
    print(f"Total download time: {report['download_time'] / 60:.2f} minutes")
    
    return report

# Define the DAG
with DAG(
    'huggingface_dataset_downloads',
    default_args=default_args,
    description='Download datasets from HuggingFace',
    schedule_interval=timedelta(days=7),  # Weekly execution
    start_date=datetime(2025, 4, 1),
    catchup=False,
    tags=['huggingface', 'llm', 'dataset_download'],
) as dag:
    
    # Start the pipeline
    start_task = DummyOperator(
        task_id='start_download_pipeline',
    )
    
    # Create directories
    setup_dirs = PythonOperator(
        task_id='setup_download_directories',
        python_callable=create_download_directories,
        provide_context=True,
    )
    
    # Dataset Download Task Group
    with TaskGroup(group_id='dataset_downloads') as dataset_downloads:
        config = get_download_config()
        
        # Create a task for each dataset to download
        download_tasks = []
        for dataset_config in config["datasets_to_download"]:
            dataset_id = dataset_config["id"]
            subset = dataset_config.get("subset")
            split = dataset_config.get("split")
            revision = dataset_config.get("revision")
            
            # Create sanitized ID for task naming
            task_id_parts = [dataset_id.replace('/', '_')]
            if subset:
                task_id_parts.append(subset)
            if split:
                task_id_parts.append(split)
            task_id = '_'.join(task_id_parts)
            
            # Define the output directory
            output_dir = os.path.join(
                config["download_dir"],
                dataset_id.replace('/', '_'),
                subset or "default",
                split or "all"
            )
            
            # Add download task
            download_task = HuggingFaceDownloadOperator(
                task_id=f'download_{task_id}',
                dataset_id=dataset_id,
                subset=subset,
                split=split,
                revision=revision,
                output_dir=output_dir,
                api_key=config["api_key"],
                cache_dir=config["cache_dir"],
                rate_limit=config["rate_limit"],
                retry=config["retry"],
                timeout=config["timeout"]
            )
            download_tasks.append(download_task)
    
    # Generate download report
    report_task = PythonOperator(
        task_id='generate_download_report',
        python_callable=generate_download_report,
        provide_context=True,
    )
    
    # End the pipeline
    end_task = DummyOperator(
        task_id='end_download_pipeline',
    )
    
    # Set up task dependencies
    start_task >> setup_dirs >> dataset_downloads >> report_task >> end_task