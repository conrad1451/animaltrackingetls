# CHQ: Claude AI generated file

"""
retry_config.py
---------------
Shared tenacity retry settings.  Import the pre-built decorator so every
HTTP-calling module uses the same back-off policy without copy-pasting.

Usage
-----
    from retry_config import http_retry

    @http_retry
    def my_api_call(...):
        ...
"""

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

#: Decorator for any function that makes an outbound HTTP request.
#: Retries up to 5 times with exponential back-off (2 s → 10 s ceiling).
http_retry = retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
    )),
    reraise=True,
)