from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/cosmopedia", "stories")

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional
from datetime import datetime


class BaseConnector(ABC):
    """
    Abstract base class for all data connectors in the MLOps pipeline.
    Provides a standard interface for connecting to data sources,
    fetching data, transforming it, and logging metadata.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the connector with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters for the connector
        """
        self.config = config
        self.connection = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.metadata = {
            "connector_type": self.__class__.__name__,
            "connection_attempts": 0,
            "last_fetch_timestamp": None,
            "records_processed": 0,
            "errors": []
        }
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def fetch_data(self, query_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve data from the connected source.
        
        Args:
            query_params: Optional parameters to customize the data fetch
            
        Returns:
            Dict containing the fetched data
        """
        pass
    
    @abstractmethod
    def transform_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw data into the standardized format required by the MLOps pipeline.
        
        Args:
            raw_data: The data as fetched from the source
            
        Returns:
            Dict containing the transformed data
        """
        pass
    
    def log_metadata(self) -> Dict[str, Any]:
        """
        Record and return metadata about the connector operations.
        
        Returns:
            Dict containing metadata about the connector's operations
        """
        self.metadata["last_logged"] = datetime.now().isoformat()
        self.logger.info(f"Connector ***REMOVED*** {self.metadata}")
        return self.metadata
    
    def close(self) -> None:
        """
        Close the connection and perform any necessary cleanup.
        """
        self.connection = None
        self.logger.info("Connection closed")
    
    def __enter__(self):
        """
        Support for context manager protocol.
        """
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Support for context manager protocol.
        """
        self.close()
    