"""
HuggingFace Airflow Operators

Custom Airflow operators for interacting with HuggingFace models and datasets
while applying rate limiting, retry logic, and proper error handling.
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional, List

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException

# Import our connectors
from connectors.hf_dataset_connector import HuggingFaceDatasetConnector

class HuggingFaceDownloadOperator(BaseOperator):
    """
    Operator for downloading datasets from HuggingFace.
    
    Attributes:
        dataset_id: ID of the dataset to download
        subset: Optional subset of the dataset
        split: Optional split of the dataset
        output_dir: Directory to save the downloaded dataset
        api_key: HuggingFace API key
        cache_dir: Directory to cache downloaded datasets
        rate_limit: Dict with rate limit settings
        retry: Dict with retry settings
    """
    
    @apply_defaults
    def __init__(
        self,
        dataset_id: str,
        output_dir: str,
        api_key: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: str = "./dataset_cache",
        rate_limit: Optional[Dict[str, Any]] = None,
        retry: Optional[Dict[str, Any]] = None,
        timeout: float = 3600.0,  # 1 hour default timeout
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dataset_id = dataset_id
        self.subset = subset
        self.split = split
        self.revision = revision
        self.output_dir = output_dir
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit or {}
        self.retry = retry or {}
        self.timeout = timeout
    
    def execute(self, context):
        """
        Execute the operator.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dict containing download status
        """
        self.log.info(f"Downloading dataset: {self.dataset_id}")
        if self.subset:
            self.log.info(f"Subset: {self.subset}")
        if self.split:
            self.log.info(f"Split: {self.split}")
        
        # Create configuration for the connector
        config = {
            "api_key": self.api_key,
            "cache_dir": self.cache_dir,
            "download_dir": self.output_dir,
            "rate_limit": self.rate_limit,
            "retry": self.retry,
            # Not using queue for Airflow tasks
            "use_queue": False
        }
        
        try:
            # Initialize connector
            connector = HuggingFaceDatasetConnector(config)
            
            # Connect to HuggingFace API
            if not connector.connect():
                raise AirflowException(f"Failed to connect to HuggingFace API for dataset {self.dataset_id}")
            
            # Download the dataset
            download_result = connector.download_dataset(
                dataset_id=self.dataset_id,
                subset=self.subset,
                split=self.split,
                revision=self.revision,
                target_dir=self.output_dir,
                wait=True,  # Wait for completion
                timeout=self.timeout
            )
            
            if "error" in download_result:
                raise AirflowException(f"Error downloading dataset: {download_result['error']}")
            
            # Add execution metadata
            download_result["airflow_execution_date"] = context['execution_date'].isoformat()
            download_result["airflow_dag_id"] = context['dag'].dag_id
            download_result["airflow_task_id"] = context['task'].task_id
            
            # Save download info
            info_path = os.path.join(self.output_dir, f"{self.dataset_id.replace('/', '_')}_download_info.json")
            with open(info_path, 'w') as f:
                json.dump(download_result, f, indent=2)
            
            # Log success
            self.log.info(f"Dataset downloaded successfully: {self.dataset_id}")
            self.log.info(f"Number of rows: {download_result.get('num_rows', 'unknown')}")
            self.log.info(f"Download time: {download_result.get('duration', 0):.2f} seconds")
            
            # Close the connector
            connector.close()
            
            return download_result
            
        except Exception as e:
            self.log.exception(f"Error in HuggingFaceDownloadOperator for {self.dataset_id}: {str(e)}")
            raise AirflowException(f"Failed to download dataset: {str(e)}")