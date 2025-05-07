import os
import time
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import requests
import sys
import os
import queue
import threading
import time
import logging
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional, Tuple
from dataclasses import dataclass
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import login

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the base connector
from connectors.base_connector_interface import BaseConnector

# Import our utility modules
from utils.rate_limiting_utils import TokenBucket, with_retry
from utils.queue import LLMRequestQueue, LLMRequest

# For dataset handling
from dataclasses import dataclass

@dataclass
class DatasetRequest:
    """Data class representing a dataset download request."""
    id: str
    dataset_id: str
    subset: Optional[str]
    split: Optional[str]
    revision: Optional[str]
    target_dir: str
    timestamp: float
    priority: int = 0
    callback: Optional[Callable] = None
    metadata: Optional[Dict[str, Any]] = None

class HuggingFaceDatasetConnector(BaseConnector):
    """
    Connector for downloading and processing datasets from HuggingFace Hub.
    Includes rate limiting, retry logic, and queueing for large dataset operations.
    """
    
    # Class-level rate limiter for all instances to share
    # Default: 5 requests per second with max burst of 10
    _global_rate_limiter = TokenBucket(tokens_per_second=5, max_tokens=10)
    
    # Shared connection pool
    _session = None
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace dataset connector.
        
        Args:
            config: Dictionary with configuration parameters including:
                - api_key: HuggingFace API key
                - cache_dir: Directory to cache downloaded datasets
                - rate_limit: Dict with rate limit settings
                - retry: Dict with retry settings
                - use_queue: Whether to use the request queue
                - queue_config: Dict with queue settings
                - download_dir: Directory to save downloaded datasets
        """
        super().__init__(config)
        
        # Extract configuration with defaults
        self.api_key = config.get("api_key")
        self.cache_dir = config.get("cache_dir", "./dataset_cache")
        self.download_dir = config.get("download_dir", "./datasets")
        self.base_url = "https://huggingface.co/api/datasets"
        self.hub_url = "https://huggingface.co/datasets"
        
        # Rate limiting config
        rate_config = config.get("rate_limit", {})
        self.instance_rate_limiter = TokenBucket(
            tokens_per_second=rate_config.get("tokens_per_second", 2),
            max_tokens=rate_config.get("max_tokens", 5)
        )
        
        # Retry config
        self.retry_config = config.get("retry", {
            "max_retries": 5,
            "base_delay": 1.0,
            "max_delay": 60  # Longer for dataset operations
        })
        
        # Dataset download queue
        self.use_queue = config.get("use_queue", True)
        self.queue = None
        if self.use_queue:
            queue_config = config.get("queue_config", {})
            self.queue = DatasetDownloadQueue(
                max_size=queue_config.get("max_size", 100),
                num_workers=queue_config.get("num_workers", 2),
                request_timeout=queue_config.get("request_timeout", 3600.0)  # 1 hour for large datasets
            )
            # Set the processor function
            self.queue.set_processor(self._process_queued_download)
            # Start queue workers
            self.queue.start()
        
        # Create necessary directories
        for directory in [self.cache_dir, self.download_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        # Initialize shared request session if not already done
        if HuggingFaceDatasetConnector._session is None:
            HuggingFaceDatasetConnector._session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=100,
                max_retries=0,  # We'll handle retries ourselves
                pool_block=False
            )
            HuggingFaceDatasetConnector._session.mount('https://', adapter)
    
    def connect(self) -> bool:
        """
        Establish connection to the HuggingFace API with rate limiting and retry.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        self.metadata["connection_attempts"] += 1
        
        if not self.api_key:
            self.logger.warning("No API key provided, proceeding with unauthenticated access")
            self.connection = {
                "headers": {},
                "connected_at": time.time(),
                "authenticated": False
            }
            return True
        
        try:
            # Wait for rate limit
            if not self._global_rate_limiter.take():
                self.logger.info("Global rate limit hit, waiting...")
                self._global_rate_limiter.wait_for_tokens(timeout=5.0)
            
            # Test connection with a simple API call
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = self._make_api_request(
                method="GET",
                endpoint="/HuggingFaceTB/cosmopedia",  # Example of a simple API call
                description="connection test"
            )
            
            if response.status_code in [200, 401, 403]:  # Even auth failures indicate the API is reachable
                authenticated = response.status_code == 200
                self.connection = {
                    "headers": headers if authenticated else {},
                    "connected_at": time.time(),
                    "authenticated": authenticated
                }
                # Set up authentication for the datasets library
                if authenticated:
                    try:
                        login(token=self.api_key)
                        self.logger.info("Successfully set up authentication for datasets library")
                    except Exception as e:
                        self.logger.warning(f"Failed to set up datasets library authentication: {str(e)}")
                        # Continue anyway, as direct API access is authenticated
                if not authenticated:
                    self.logger.warning("Connected to API but authentication failed")
                else:
                    self.logger.info("Successfully connected to HuggingFace API")
                return True
            else:
                self.logger.error(f"Failed to connect: {response.status_code} - {response.text}")
                self.metadata["errors"].append(f"Connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.exception(f"Exception during connection: {str(e)}")
            self.metadata["errors"].append(f"Connection exception: {str(e)}")
            return False
    
    @with_retry()  # Use default retry parameters
    def _make_api_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        description: str = "API request"
    ) -> requests.Response:
        """
        Make an API request with rate limiting and retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            params: Query parameters
            json_data: JSON data for POST requests
            description: Description for logging
            
        Returns:
            requests.Response object
        """
        # Apply instance-specific rate limiting
        if not self.instance_rate_limiter.take():
            self.logger.debug(f"Instance rate limit hit for {description}, waiting...")
            self.instance_rate_limiter.wait_for_tokens(timeout=10.0)
        
        # Apply global rate limiting
        if not self._global_rate_limiter.take():
            self.logger.debug(f"Global rate limit hit for {description}, waiting...")
            self._global_rate_limiter.wait_for_tokens(timeout=5.0)
        
        # Log the request
        self.logger.debug(f"Making {method} request to {endpoint}")
        
        # Make the request
        url = f"{self.base_url}{endpoint}"
        headers = {}
        if self.connection and self.connection.get("authenticated", False):
            headers = self.connection["headers"]
        
        return self._session.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            headers=headers,
            timeout=(5.0, 180)  # Longer timeouts for dataset operations
        )
    
    def fetch_data(self, dataset_id: str, query_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch dataset metadata from HuggingFace.
        
        Args:
            dataset_id: ID of the dataset to fetch
            query_params: Optional parameters to customize the API request
        
        Returns:
            Dict containing dataset metadata
        """
        if not self.connection:
            success = self.connect()
            if not success:
                self.logger.error("Cannot fetch data: Failed to connect")
                return {"error": "Connection failed"}
            
        query_params = query_params or {}
        
        try:
            # Fetch dataset info
            endpoint = f"/{dataset_id}"
            response = self._make_api_request(
                method="GET",
                endpoint=endpoint,
                params=query_params,
                description=f"fetch dataset info for {dataset_id}"
            )
            
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch dataset data: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}"}
            
            dataset_data = response.json()
            
            self.metadata["last_fetch_timestamp"] = time.time()
            self.logger.info(f"Successfully fetched metadata for dataset {dataset_id}")
            
            return dataset_data
            
        except Exception as e:
            self.logger.exception(f"Exception during data fetch: {str(e)}")
            self.metadata["errors"].append(f"Fetch exception: {str(e)}")
            return {"error": str(e)}
    
    def transform_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw HuggingFace dataset data into standardized format.
        
        Args:
            raw_data: The data as fetched from HuggingFace
            
        Returns:
            Dict containing the transformed dataset data
        """
        if "error" in raw_data:
            return raw_data
        
        try:
            # Extract and standardize the dataset metadata
            transformed = {
                "dataset_id": raw_data.get("id", ""),
                "name": raw_data.get("name", ""),
                "description": raw_data.get("description", ""),
                "citation": raw_data.get("citation", ""),
                "license": raw_data.get("license", ""),
                "tags": raw_data.get("tags", []),
                "downloads": raw_data.get("downloads", 0),
                "author": raw_data.get("author", ""),
                "subsets": self._extract_subsets(raw_data),
                "languages": raw_data.get("languages", []),
                "size_categories": raw_data.get("size_categories", []),
                "features": self._extract_features(raw_data),
                "last_modified": raw_data.get("lastModified", ""),
                "splits": self._extract_splits(raw_data),
                "raw_metadata": raw_data if self.config.get("include_raw", False) else None
            }
            
            self.metadata["records_processed"] += 1
            return transformed
            
        except Exception as e:
            self.logger.exception(f"Exception during data transformation: {str(e)}")
            self.metadata["errors"].append(f"Transform exception: {str(e)}")
            return {"error": f"Transformation failed: {str(e)}"}
    
    def _extract_subsets(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and standardize dataset subset information."""
        subsets = []
        if "subsets" in raw_data and isinstance(raw_data["subsets"], list):
            for subset in raw_data["subsets"]:
                subsets.append({
                    "name": subset.get("name", ""),
                    "description": subset.get("description", ""),
                    "splits": subset.get("splits", [])
                })
        return subsets
    
    def _extract_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and standardize dataset feature information."""
        features = {}
        if "features" in raw_data and isinstance(raw_data["features"], list):
            for feature in raw_data["features"]:
                feature_name = feature.get("name", "")
                features[feature_name] = {
                    "type": feature.get("type", ""),
                    "description": feature.get("description", "")
                }
        return features
    
    def _extract_splits(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and standardize dataset split information."""
        splits = {}
        if "splits" in raw_data and isinstance(raw_data["splits"], list):
            for split in raw_data["splits"]:
                split_name = split.get("name", "")
                splits[split_name] = {
                    "num_examples": split.get("num_examples", 0),
                    "num_bytes": split.get("num_bytes", 0)
                }
        return splits
    
    def list_datasets(
        self, 
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available datasets with optional filtering.
        
        Args:
            filter_params: Parameters to filter results, such as:
                - author: Filter by dataset author
                - search: Search term in dataset name/description
                - tags: List of tags to filter by
                - sort: Sort field (downloads, likes, etc.)
                - direction: Sort direction (asc, desc)
                - limit: Maximum number of results
                - offset: Pagination offset
                
        Returns:
            List of dataset metadata dictionaries
        """
        if not self.connection:
            success = self.connect()
            if not success:
                self.logger.error("Cannot list datasets: Failed to connect")
                return [{"error": "Connection failed"}]
        
        filter_params = filter_params or {}
        
        try:
            response = self._make_api_request(
                method="GET", 
                endpoint="", 
                params=filter_params,
                description="list datasets"
            )
            
            if response.status_code != 200:
                self.logger.error(f"Failed to list datasets: {response.status_code} - {response.text}")
                return [{"error": f"API error: {response.status_code}"}]
            
            results = response.json()
            
            # Transform results to standard format
            datasets = []
            for item in results.get("items", []):
                datasets.append({
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                    "description": item.get("description", ""),
                    "tags": item.get("tags", []),
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "author": item.get("author", "")
                })
            
            self.logger.info(f"Listed {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            self.logger.exception(f"Exception during dataset listing: {str(e)}")
            return [{"error": str(e)}]
    
    def download_dataset(
        self,
        dataset_id: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        revision: Optional[str] = None,
        target_dir: Optional[str] = None,
        wait: bool = True,
        callback: Optional[Callable] = None,
        timeout: float = 3600.0  # 1 hour default timeout for large datasets
    ) -> Dict[str, Any]:
        """
        Download a dataset with rate limiting and optional queueing.
        
        Args:
            dataset_id: ID of the dataset to download
            subset: Optional subset of the dataset
            split: Optional split of the dataset
            revision: Optional revision/version of the dataset
            target_dir: Directory to save the dataset (defaults to download_dir)
            wait: Whether to wait for the download to complete
            callback: Optional callback function for async processing
            timeout: Maximum time to wait for download in seconds
            
        Returns:
            Dict containing download status and path information
        """
        # Prepare download parameters
        target_path = target_dir or os.path.join(self.download_dir, dataset_id.replace("/", "_"))
        os.makedirs(target_path, exist_ok=True)
        
        # If using queue and not waiting, add to queue and return request ID
        if self.use_queue and not wait:
            # Use the request ID returned by add_request
            request_id = self.queue.add_request(
                dataset_id=dataset_id,
                subset=subset,
                split=split,
                revision=revision,
                target_dir=target_path,
                callback=callback
            )
            return {
                "request_id": request_id,
                "queued": True,
                "dataset_id": dataset_id,
                "target_path": target_path
            }
        
        # If using queue and waiting, add to queue and wait for result
        elif self.use_queue and wait:
            # Use the request ID returned by add_request
            request_id = self.queue.add_request(
                dataset_id=dataset_id,
                subset=subset,
                split=split,
                revision=revision,
                target_dir=target_path
            )
            try:
                result = self.queue.get_result(request_id, timeout=timeout)
                return result
            except TimeoutError:
                return {
                    "error": "Download request timed out",
                    "request_id": request_id,
                    "dataset_id": dataset_id
                }
        
        # If not using queue, process directly
        else:
            return self._download_dataset_direct(
                dataset_id=dataset_id,
                subset=subset,
                split=split,
                revision=revision,
                target_dir=target_path
            )
    
    @with_retry(max_retries=5, base_delay=2.0, max_delay=60.0)
    def _download_dataset_direct(
        self,
        dataset_id: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        revision: Optional[str] = None,
        target_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download a dataset directly with retries.
        
        Args:
            dataset_id: ID of the dataset to download
            subset: Optional subset of the dataset
            split: Optional split of the dataset
            revision: Optional revision/version of the dataset
            target_dir: Directory to save the dataset
            
        Returns:
            Dict containing download status and path information
        """
        try:
            # Use the datasets library for efficient downloading
            
            start_time = time.time()
            self.logger.info(f"Starting download of dataset {dataset_id}")
            
            # Get available configs if no subset is specified
            if not subset:
                try:
                    configs = get_dataset_config_names(dataset_id)
                    if configs and len(configs) > 0:
                        subset = configs[0]  # Use first config as default
                        self.logger.info(f"No subset specified, using first available: {subset}")
                except Exception as e:
                    self.logger.warning(f"Failed to get config names: {str(e)}")
                    # If no configs found, proceed without subset
                    pass
            
            # Create download tracking info
            download_info = {
                "dataset_id": dataset_id,
                "subset": subset,
                "split": split,
                "revision": revision,
                "target_dir": target_dir,
                "start_time": start_time,
                "status": "in_progress"
            }
            
            # Prepare download arguments
            load_args = {
                "path": dataset_id,
                
            }
            if subset:
                load_args["name"] = subset
            if revision:
                load_args["revision"] = revision
            
            # Load the dataset (this will download it if not cached)
            dataset = load_dataset(**load_args)
            
            # Get specific split if requested
            if split and split in dataset:
                dataset = dataset[split]
            
            # Save dataset to target directory
            target_path = os.path.join(target_dir, f"{dataset_id.replace('/', '_')}")
            if subset:
                target_path += f"_{subset}"
            if split:
                target_path += f"_{split}"
            
            # Save in Arrow format for efficiency
            arrow_path = target_path + ".arrow"
            dataset.save_to_disk(arrow_path)
            
            # Save a small JSON sample for preview
            json_sample_path = target_path + "_sample.json"
            sample_size = min(10, len(dataset))
            with open(json_sample_path, 'w') as f:
                json.dump(dataset[:sample_size], f, indent=2)
            
            # Save dataset info
            info_path = target_path + "_info.json"
            with open(info_path, 'w') as f:
                # Convert dataset info to dict and save
                info_dict = {
                    "features": str(dataset.features),
                    "shape": dataset.shape,
                    "num_rows": dataset.num_rows,
                    "dataset_id": dataset_id,
                    "subset": subset,
                    "split": split,
                    "revision": revision,
                    "download_time": time.time() - start_time
                }
                json.dump(info_dict, f, indent=2)
            
            duration = time.time() - start_time
            self.logger.info(f"Dataset {dataset_id} downloaded successfully in {duration:.2f}s")
            
            # Update and return download info
            download_info.update({
                "status": "completed",
                "duration": duration,
                "file_paths": {
                    "arrow": arrow_path,
                    "sample": json_sample_path,
                    "info": info_path
                },
                "num_rows": dataset.num_rows,
                "size_bytes": sum(os.path.getsize(p) for p in [arrow_path, json_sample_path, info_path])
            })
            
            return download_info
            
        except Exception as e:
            self.logger.exception(f"Exception during dataset download: {str(e)}")
            self.logger.exception(f"Failed to download dataset {dataset_id}: {str(e)}")
            return {
                "error": str(e),
                "dataset_id": dataset_id,
                "subset": subset,
                "status": "failed",
                "duration": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def _process_queued_download(self, request: DatasetRequest) -> Dict[str, Any]:
        """
        Process a queued dataset download request.
        
        Args:
            request: DatasetRequest object from the queue
            
        Returns:
            Dict containing the download result
        """
        self.logger.info(f"Processing queued download {request.id} for dataset {request.dataset_id}")
        
        # Run the download
        result = self._download_dataset_direct(
            dataset_id=request.dataset_id,
            subset=request.subset,
            split=request.split,
            revision=request.revision,
            target_dir=request.target_dir
        )
        
        # Add request metadata to result
        result["request_id"] = request.id
        result["queued_at"] = request.timestamp
        result["queue_time"] = time.time() - request.timestamp
        
        return result
    
    def close(self) -> None:
        """Close the connection and stop the queue."""
        super().close()
        
        # Stop the queue if it was started
        if self.queue is not None:
            self.queue.stop()
            self.logger.info("Dataset download queue stopped")


class DatasetDownloadQueue:
    """
    Queue system for handling dataset download requests.
    Similar to LLMRequestQueue but specialized for dataset operations.
    """
    
    def __init__(
        self, 
        max_size: int = 100,
        num_workers: int = 2,
        request_timeout: float = 3600.0
    ):
        """
        Initialize the dataset download queue.
        
        Args:
            max_size: Maximum queue size
            num_workers: Number of worker threads
            request_timeout: Timeout for requests in seconds
        """
        # Use PriorityQueue for priority-based processing
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.num_workers = num_workers
        self.request_timeout = request_timeout
        self.workers = []
        self.running = False
        self.results = {}
        self.results_lock = threading.Lock()
        self.result_event = threading.Event()
        self.processor = None
        self.logger = logging.getLogger("dataset_queue")
        
    def set_processor(self, processor_func: Callable[[DatasetRequest], Dict[str, Any]]):
        """Set the function that processes queue items."""
        self.processor = processor_func
        
    def start(self):
        """Start the worker threads."""
        if self.running:
            return
            
        if not self.processor:
            raise ValueError("Processor function must be set before starting the queue")
            
        self.running = True
        self.workers = []
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"DatasetQueueWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        self.logger.info(f"Started {self.num_workers} dataset queue workers")
        
    def stop(self):
        """Stop the worker threads."""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=2.0)
                
        self.logger.info("Stopped dataset queue workers")
        
    def add_request(
        self,
        dataset_id: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        revision: Optional[str] = None,
        target_dir: Optional[str] = None,
        priority: int = 0,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a dataset download request to the queue."""
        request_id = str(time.time()) + "_" + dataset_id.replace("/", "_")
        timestamp = time.time()
        
        request = DatasetRequest(
            id=request_id,
            dataset_id=dataset_id,
            subset=subset,
            split=split,
            revision=revision,
            target_dir=target_dir or "",
            timestamp=timestamp,
            priority=priority,
            callback=callback,
            metadata=metadata or {}
        )
        
        # Add to queue with priority
        try:
            self.queue.put((priority, timestamp, request), block=True, timeout=1.0)
            self.logger.debug(f"Added request {request_id} to queue (priority {priority})")
            return request_id  # Return the actual request ID used internally
        except queue.Full:
            self.logger.warning("Queue is full, request rejected")
            raise RuntimeError("Dataset download queue is full")
            
    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get the result for a specific request ID."""
        start_time = time.time()
        
        while timeout is None or time.time() - start_time < timeout:
            with self.results_lock:
                if request_id in self.results:
                    return self.results[request_id]
            
            # Wait for a new result to be available
            if not self.result_event.wait(timeout=0.1):
                continue
            self.result_event.clear()
        
        raise TimeoutError(f"Timeout waiting for result of request {request_id}")
        
    def _worker_loop(self):
        """Worker thread function for processing queue items."""
        while self.running:
            try:
                # Get next request from queue
                _, _, request = self.queue.get(block=True, timeout=0.5)
                
                # Check if request is too old
                if time.time() - request.timestamp > self.request_timeout:
                    self.logger.warning(f"Request {request.id} timed out in queue")
                    self.queue.task_done()
                    continue
                
                try:
                    self.logger.debug(f"Processing request {request.id}")
                    result = self.processor(request)
                    
                    # Store result
                    with self.results_lock:
                        self.results[request.id] = result
                    
                    # Signal that a new result is available
                    self.result_event.set()
                    
                    # Call callback if provided
                    if request.callback:
                        try:
                            request.callback(result)
                        except Exception as e:
                            self.logger.error(f"Error in callback for request {request.id}: {str(e)}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing request {request.id}: {str(e)}")
                    with self.results_lock:
                        self.results[request.id] = {
                            "error": str(e),
                            "timestamp": time.time(),
                            "dataset_id": request.dataset_id,
                            "status": "failed"
                        }
                    self.result_event.set()
                
                finally:
                    self.queue.task_done()
                    
            except queue.Empty:
                # Queue is empty, just continue
                pass
            except Exception as e:
                self.logger.error(f"Unexpected error in dataset queue worker: {str(e)}")
                time.sleep(1)  # Avoid tight loop in case of recurring errors