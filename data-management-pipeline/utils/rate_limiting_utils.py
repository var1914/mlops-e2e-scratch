import time
import random
import functools
import logging
import requests
from typing import Callable, Any, Dict, Optional, TypeVar, List
from threading import Lock, Timer
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("rate_limiting")

# Type variable for function return types
T = TypeVar('T')

class TokenBucket:
    """
    Token bucket rate limiter implementation.
    Allows for bursts of requests up to a maximum number of tokens,
    then throttles requests to a specific rate.
    """
    def __init__(self, tokens_per_second: float, max_tokens: int):
        """
        Initialize token bucket.
        
        Args:
            tokens_per_second: Rate at which tokens refill
            max_tokens: Maximum number of tokens the bucket can hold
        """
        self.tokens_per_second = tokens_per_second
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_refill = time.time()
        self.lock = Lock()
        
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.tokens_per_second
        
        with self.lock:
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.last_refill = now
    
    def take(self, tokens: int = 1) -> bool:
        """
        Take tokens from the bucket if available.
        
        Args:
            tokens: Number of tokens to take
            
        Returns:
            bool: True if tokens were successfully taken, False otherwise
        """
        self._refill()
        
        with self.lock:
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Wait until tokens are available and take them.
        
        Args:
            tokens: Number of tokens to take
            timeout: Maximum time to wait in seconds, or None for no timeout
            
        Returns:
            bool: True if tokens were successfully taken, False if timed out
        """
        start_time = time.time()
        
        while True:
            if self.take(tokens):
                return True
            
            if timeout is not None and time.time() - start_time > timeout:
                return False
            
            # Sleep for a short time before trying again
            # Calculate sleep time based on when we expect tokens to be available
            with self.lock:
                tokens_needed = tokens - self.tokens
                if tokens_needed <= 0:
                    continue
                wait_time = tokens_needed / self.tokens_per_second
                # Add a small buffer to avoid busy waiting
                wait_time = min(wait_time, 0.1)
            
            time.sleep(wait_time)


def with_retry(
    max_retries: int = 3, 
    base_delay: float = 0.1, 
    max_delay: float = 30, 
    jitter: bool = True,
    retryable_exceptions: List[type] = None,
    retryable_status_codes: List[int] = None
) -> Callable:
    """
    Decorator for implementing exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        jitter: Whether to add random jitter to delay
        retryable_exceptions: List of exception types that should trigger retry
        retryable_status_codes: List of HTTP status codes that should trigger retry
        
    Returns:
        Decorator function
    """
    if retryable_exceptions is None:
        retryable_exceptions = [
            ConnectionError, 
            TimeoutError,
            requests.RequestException
        ]
        
    if retryable_status_codes is None:
        retryable_status_codes = [429, 500, 502, 503, 504]
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            retry_count = 0
            delay = base_delay
            
            while True:
                try:
                    response = func(*args, **kwargs)
                    
                    # Handle retryable status codes for requests
                    if hasattr(response, 'status_code') and response.status_code in retryable_status_codes:
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.warning(
                                f"Max retries ({max_retries}) exceeded with status code {response.status_code}"
                            )
                            return response
                            
                        # Calculate next delay
                        next_delay = calculate_delay(delay, retry_count, max_delay, jitter)
                        
                        logger.info(
                            f"Received status code {response.status_code}, "
                            f"retrying in {next_delay:.2f}s (attempt {retry_count}/{max_retries})"
                        )
                        time.sleep(next_delay)
                        delay = next_delay
                        continue
                    
                    # No error, return the response
                    return response
                    
                except tuple(retryable_exceptions) as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.warning(f"Max retries ({max_retries}) exceeded: {str(e)}")
                        raise
                    
                    # Calculate next delay
                    next_delay = calculate_delay(delay, retry_count, max_delay, jitter)
                    
                    logger.info(
                        f"Retry attempt {retry_count}/{max_retries} after error: {str(e)}. "
                        f"Retrying in {next_delay:.2f}s"
                    )
                    time.sleep(next_delay)
                    delay = next_delay
                    
        return wrapper
    
    return decorator


def calculate_delay(current_delay: float, retry_count: int, max_delay: float, add_jitter: bool) -> float:
    """
    Calculate the next retry delay using exponential backoff.
    
    Args:
        current_delay: Current delay in seconds
        retry_count: Current retry attempt
        max_delay: Maximum delay in seconds
        add_jitter: Whether to add random jitter
        
    Returns:
        float: Next delay in seconds
    """
    # Exponential backoff: delay = base_delay * 2^retry_count
    next_delay = current_delay * (2 ** (retry_count - 1))
    
    # Add jitter to avoid thundering herd problem
    if add_jitter:
        jitter = random.uniform(0, 0.1 * next_delay)
        next_delay += jitter
    
    # Cap at max_delay
    return min(next_delay, max_delay)


class RateLimiter:
    """
    Rate limiter that can be used to limit the number of requests per second.
    Uses TokenBucket implementation internally.
    """
    def __init__(self, requests_per_second: float, burst_size: int = None):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum number of requests per second
            burst_size: Maximum burst size (default: 2 * requests_per_second)
        """
        if burst_size is None:
            burst_size = max(1, int(2 * requests_per_second))
            
        self.bucket = TokenBucket(requests_per_second, burst_size)
        self.requests_per_second = requests_per_second
        
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to rate limit a function.
        
        Args:
            func: Function to rate limit
            
        Returns:
            Rate limited function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Wait for a token to be available
            self.bucket.wait_for_tokens(1)
            return func(*args, **kwargs)
        
        return wrapper
    
    def wait(self) -> None:
        """Wait until a request can be made according to the rate limit."""
        self.bucket.wait_for_tokens(1)


# Example usage:
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example 1: Rate limiting with the RateLimiter class
    rate_limiter = RateLimiter(requests_per_second=2)
    
    @rate_limiter
    def make_api_request(endpoint):
        logger.info(f"Making request to {endpoint}")
        # Simulate API request
        time.sleep(0.1)
        return {"status": "success", "endpoint": endpoint}
    
    # Example 2: Using with_retry decorator
    @with_retry(max_retries=3, base_delay=1.0)
    def fetch_data(url):
        logger.info(f"Fetching data from {url}")
        # Simulate request with potential failures
        if random.random() < 0.7:  # 70% chance of failure for demo purposes
            response = type('Response', (), {'status_code': random.choice([429, 500])})
            return response
        return type('Response', (), {'status_code': 200, 'data': 'Success!'})
    
    # Demo
    def run_demo():
        # Demo rate limiting
        logger.info("Starting rate limiting demo...")
        for i in range(5):
            make_api_request(f"/api/endpoint{i}")
        
        # Demo retry logic
        logger.info("\nStarting retry logic demo...")
        result = fetch_data("https://example.com/api")
        logger.info(f"Final result: {result.status_code}")
    
    run_demo()