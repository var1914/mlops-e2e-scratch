import logging
import time
import json
import os
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import our connector
from hf_dataset_connector import HuggingFaceDatasetConnector

def main():
    # 1. Configure the connector
    config = {
        "api_key": "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",  # Replace with your API key
        "cache_dir": "./dataset_cache",
        "download_dir": "./downloaded_datasets",
        
        # Rate limiting settings
        "rate_limit": {
            "tokens_per_second": 2,  # Instance-specific rate limit
            "max_tokens": 5         # Maximum burst size
        },
        
        # Retry settings
        "retry": {
            "max_retries": 5,
            "base_delay": 1.0,
            "max_delay": 60
        },
        
        # Queue settings
        "use_queue": True,
        "queue_config": {
            "max_size": 100,
            "num_workers": 2
        }
    }
    
    # 2. Create and initialize connector
    connector = HuggingFaceDatasetConnector(config)
    
    # 3. Connect to the API
    print("Connecting to HuggingFace API...")
    if not connector.connect():
        print("Failed to connect!")
        return
    
    print("Connected successfully!")
    
    # Download multiple datasets asynchronously
    print("\nStarting asynchronous downloads of multiple datasets...")
    
    # Define a callback function for when downloads complete
    def download_callback(result):
        print(f"\nDownload completed for {result.get('dataset_id')}:")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Status: {result.get('status')}")
            print(f"Duration: {result.get('duration', 0):.2f} seconds")
            print(f"Size: {result.get('size_bytes', 0) / (1024*1024):.2f} MB")
    
    # List of datasets to download
    datasets_to_download = [
        {"id": "HuggingFaceTB/cosmopedia", "subset": "stories", "split": "train"},
    ]
    
    # Start asynchronous downloads
    request_ids = []
    for dataset in datasets_to_download:
        result = connector.download_dataset(
            dataset_id=dataset["id"],
            subset=dataset.get("subset"),
            split=dataset.get("split"),
            wait=False,  # Don't wait - use the queue
            callback=download_callback
        )
        request_ids.append(result.get("request_id"))
        print(f"Queued download for {dataset['id']}, request ID: {result.get('request_id')}")
    
    # Wait for a specific download to complete
    if request_ids:
        print(f"\nWaiting for the first queued download to complete...")
        try:
            # Wait for the first download to complete
            result = connector.queue.get_result(request_ids[0], timeout=300.0)
            print(f"First download completed: {result.get('dataset_id')}")
            print(f"Status: {result.get('status')}")
            if "error" not in result:
                print(f"Size: {result.get('size_bytes', 0) / (1024*1024):.2f} MB")
        except TimeoutError:
            print("Timeout waiting for download to complete")
    
    # Get connector metadata
    print("\nConnector metadata:")
    metadata = connector.log_metadata()
    print(f"Connection attempts: {metadata.get('connection_attempts')}")
    print(f"Records processed: {metadata.get('records_processed')}")
    print(f"Last fetch: {metadata.get('last_fetch_timestamp')}")
    print(f"Errors: {len(metadata.get('errors', []))}")
    
    # 11. Close the connection
    print("\nClosing connection and stopping queue...")
    connector.close()
    print("Done!")

if __name__ == "__main__":
    main()