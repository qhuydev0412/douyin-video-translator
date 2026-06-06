"""Property-based tests for network retry logic.

Feature: douyin-video-translator, Property 2: Network Retry Logic Bounded at 3 Attempts

Validates: Requirements 1.4

Property Statement:
For any sequence of download attempts where network failures occur, the system SHALL retry
at most max_retries times. If all retries fail, it SHALL report an error. If any attempt
succeeds, it SHALL return success immediately without further retries.
Non-retryable errors (VideoNotFoundError) are raised immediately without retry.
"""

from unittest.mock import patch

import pytest
from hypothesis import given, settings, strategies as st

from app.services.downloader import (
    DownloadError,
    NetworkError,
    VideoDownloader,
    VideoNotFoundError,
)


@settings(max_examples=100)
@given(max_retries=st.integers(min_value=1, max_value=10))
def test_all_attempts_fail_raises_network_error_with_exact_call_count(max_retries: int):
    """If all N attempts fail with retryable errors, NetworkError is raised and exactly N calls are made.

    **Validates: Requirements 1.4**
    """
    call_count = 0

    def failing_operation():
        nonlocal call_count
        call_count += 1
        raise DownloadError("Network timeout", retryable=True)

    downloader = VideoDownloader(max_retries=max_retries, backoff_base=0)

    with pytest.raises(NetworkError) as exc_info:
        downloader._execute_with_retry(failing_operation)

    assert exc_info.value.max_retries == max_retries
    assert call_count == max_retries


@settings(max_examples=100)
@given(
    data=st.data(),
    max_retries=st.integers(min_value=1, max_value=10),
)
def test_success_on_attempt_k_calls_operation_exactly_k_times(data, max_retries: int):
    """If operation succeeds on attempt K (1 <= K <= N), operation is called exactly K times.

    **Validates: Requirements 1.4**
    """
    success_on_attempt = data.draw(
        st.integers(min_value=1, max_value=max_retries),
        label="success_on_attempt",
    )
    call_count = 0
    expected_result = {"status": "ok", "data": "video_info"}

    def operation_succeeds_on_k():
        nonlocal call_count
        call_count += 1
        if call_count < success_on_attempt:
            raise DownloadError("Network timeout", retryable=True)
        return expected_result

    downloader = VideoDownloader(max_retries=max_retries, backoff_base=0)

    result = downloader._execute_with_retry(operation_succeeds_on_k)

    assert result == expected_result
    assert call_count == success_on_attempt


@settings(max_examples=100)
@given(max_retries=st.integers(min_value=1, max_value=10))
def test_video_not_found_error_is_never_retried(max_retries: int):
    """VideoNotFoundError is raised immediately without retry regardless of max_retries.

    **Validates: Requirements 1.4**
    """
    call_count = 0

    def operation_raises_not_found():
        nonlocal call_count
        call_count += 1
        raise VideoNotFoundError()

    downloader = VideoDownloader(max_retries=max_retries, backoff_base=0)

    with pytest.raises(VideoNotFoundError):
        downloader._execute_with_retry(operation_raises_not_found)

    assert call_count == 1


@settings(max_examples=100)
@given(
    data=st.data(),
    max_retries=st.integers(min_value=2, max_value=10),
)
def test_no_further_attempts_after_success(data, max_retries: int):
    """After success on attempt K, no further retry attempts are made.

    **Validates: Requirements 1.4**
    """
    success_on_attempt = data.draw(
        st.integers(min_value=1, max_value=max_retries),
        label="success_on_attempt",
    )
    call_count = 0
    expected_result = {"video": "downloaded"}

    def operation():
        nonlocal call_count
        call_count += 1
        if call_count < success_on_attempt:
            raise DownloadError("Timeout", retryable=True)
        return expected_result

    downloader = VideoDownloader(max_retries=max_retries, backoff_base=0)

    result = downloader._execute_with_retry(operation)

    # Verify no further calls were made after success
    assert call_count == success_on_attempt
    assert result == expected_result
    # Confirm the remaining retries were NOT consumed
    remaining_retries = max_retries - success_on_attempt
    assert remaining_retries >= 0
