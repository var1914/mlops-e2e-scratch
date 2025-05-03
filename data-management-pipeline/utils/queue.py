import queue
import threading
import time
import uuid
import logging
from typing import Dict, Any, Callable, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("llm_queue")

@dataclass
class LLMRequest:
    """Data class representing an LLM request in the queue."""
    id: str
    prompt: str
    model_id: str
    parameters: Dict[str, Any]
    timestamp: float
    priority: int = 0
    callback: Optional[Callable] = None
    ***REMOVED*** Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMRequestQueue:
    """
    Queue system for handling LLM inference requests.
    Provides prioritization and rate limiting.
    """
    
    def __init__(
        self, 
        max_size: int = 1000,
        num_workers: int = 2,
        request_timeout: float = 60.0
    ):
        """
        Initialize the LLM request queue.
        
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
        self.results = {}  # Store results by request ID
        self.results_lock = threading.Lock()
        self.result_event = threading.Event()
        self.processor = None  # Will be set by set_processor
        
    def set_processor(self, processor_func: Callable[[LLMRequest], Dict[str, Any]]):
        """
        Set the function that processes queue items.
        
        Args:
            processor_func: Function that takes an LLMRequest and returns a result dict
        """
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
                name=f"LLMQueueWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Started {self.num_workers} LLM queue workers")
        
    def stop(self):
        """Stop the worker threads."""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=2.0)
                
        logger.info("Stopped LLM queue workers")
        
    def add_request(
        self, 
        prompt: str,
        model_id: str,
        parameters: Dict[str, Any],
        priority: int = 0,
        callback: Optional[Callable] = None,
        ***REMOVED*** Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a request to the queue.
        
        Args:
            prompt: Text prompt for the LLM
            model_id: Model identifier
            parameters: Model parameters
            priority: Request priority (lower number = higher priority)
            callback: Optional callback function for async processing
            ***REMOVED*** Optional metadata for the request
            
        Returns:
            str: Request ID
        """
        request_id = str(uuid.uuid4())
        timestamp = time.time()
        
        request = LLMRequest(
            id=request_id,
            prompt=prompt,
            model_id=model_id,
            parameters=parameters,
            timestamp=timestamp,
            priority=priority,
            callback=callback,
            metadata=metadata or {}
        )
        
        # Add to queue with priority
        try:
            self.queue.put((priority, timestamp, request), block=True, timeout=1.0)
            logger.debug(f"Added request {request_id} to queue (priority {priority})")
            return request_id
        except queue.Full:
            logger.warning("Queue is full, request rejected")
            raise RuntimeError("LLM request queue is full")
            
    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get the result for a specific request ID.
        Blocks until the result is available or timeout.
        
        Args:
            request_id: Request ID to get results for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dict containing the result
            
        Raises:
            TimeoutError: If timeout is reached before result is available
            KeyError: If request ID is not found
        """
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
                    logger.warning(f"Request {request.id} timed out in queue")
                    self.queue.task_done()
                    continue
                
                try:
                    logger.debug(f"Processing request {request.id}")
                    result = self.processor(request)
                    
                    # Store result
                    with self.results_lock:
                        self.results[request.id] = result
                        # Limit results cache size
                        if len(self.results) > 1000:
                            oldest = min(self.results.items(), key=lambda x: x[1].get("timestamp", 0))
                            del self.results[oldest[0]]
                    
                    # Signal that a new result is available
                    self.result_event.set()
                    
                    # Call callback if provided
                    if request.callback:
                        try:
                            request.callback(result)
                        except Exception as e:
                            logger.error(f"Error in callback for request {request.id}: {str(e)}")
                            
                except Exception as e:
                    logger.error(f"Error processing request {request.id}: {str(e)}")
                    with self.results_lock:
                        self.results[request.id] = {
                            "error": str(e),
                            "timestamp": time.time()
                        }
                    self.result_event.set()
                
                finally:
                    self.queue.task_done()
                    
            except queue.Empty:
                # Queue is empty, just continue
                pass
            except Exception as e:
                logger.error(f"Unexpected error in LLM queue worker: {str(e)}")
                time.sleep(1)  # Avoid tight loop in case of recurring errors