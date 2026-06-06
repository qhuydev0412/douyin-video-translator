"""Property-based tests for URL validation.

Feature: douyin-video-translator, Property 1: URL Validation — Only Douyin URLs Accepted

Validates: Requirements 1.2

For any URL string, the URL validation function SHALL accept it if and only if it belongs
to the douyin.com domain (including subdomains like www.douyin.com, v.douyin.com).
All other URLs SHALL be rejected.
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.services.downloader import VideoDownloader


# --- Strategies ---

# Valid Douyin subdomains
DOUYIN_SUBDOMAINS = st.sampled_from([
    "douyin.com",
    "www.douyin.com",
    "v.douyin.com",
    "live.douyin.com",
    "creator.douyin.com",
    "m.douyin.com",
    "open.douyin.com",
    "abc.douyin.com",
    "sub1.sub2.douyin.com",
])

# Valid URL schemes
VALID_SCHEMES = st.sampled_from(["http", "https"])

# Path segments
PATH_SEGMENT = st.from_regex(r"[a-zA-Z0-9_\-\.]{0,20}", fullmatch=True)

# Generate valid paths like /video/123456 or /share/abc
VALID_PATHS = st.lists(PATH_SEGMENT.filter(lambda s: len(s) > 0), min_size=0, max_size=4).map(
    lambda parts: "/" + "/".join(parts) if parts else ""
)

# Strategy for valid Douyin URLs
valid_douyin_urls = st.builds(
    lambda scheme, host, path: f"{scheme}://{host}{path}",
    scheme=VALID_SCHEMES,
    host=DOUYIN_SUBDOMAINS,
    path=VALID_PATHS,
)

# Non-Douyin domains
NON_DOUYIN_DOMAINS = st.sampled_from([
    "google.com",
    "youtube.com",
    "tiktok.com",
    "facebook.com",
    "twitter.com",
    "example.com",
    "bilibili.com",
    "weibo.com",
    "baidu.com",
    "reddit.com",
    "www.google.com",
    "v.tiktok.com",
])

# Strategy for non-Douyin URLs with valid structure
non_douyin_urls = st.builds(
    lambda scheme, host, path: f"{scheme}://{host}{path}",
    scheme=VALID_SCHEMES,
    host=NON_DOUYIN_DOMAINS,
    path=VALID_PATHS,
)

# Strategy for tricky domains that look like douyin.com but aren't
TRICKY_DOMAINS = st.sampled_from([
    "notdouyin.com",
    "douyin.com.evil.com",
    "fakedouyin.com",
    "douyin.org",
    "douyin.net",
    "douyin.cn",
    "mydouyin.com",
    "douyin.com.cn",
    "evil-douyin.com",
    "xdouyin.com",
    "douyin.comedy",
])

tricky_domain_urls = st.builds(
    lambda scheme, host, path: f"{scheme}://{host}{path}",
    scheme=VALID_SCHEMES,
    host=TRICKY_DOMAINS,
    path=VALID_PATHS,
)

# Strategy for malformed strings (not URLs at all)
malformed_strings = st.one_of(
    st.text(min_size=0, max_size=100),  # Random text
    st.just(""),  # Empty string
    st.just("   "),  # Whitespace only
    st.just("not-a-url"),
    st.just("://missing-scheme.com"),
    st.just("http://"),  # Scheme only
    st.just("douyin.com"),  # No scheme
    st.just("ftp://douyin.com"),  # Non-http scheme (still valid though)
)

# Strategy for URLs with douyin.com in path/query but not hostname
douyin_in_path_urls = st.builds(
    lambda scheme, host, suffix: f"{scheme}://{host}/redirect?url=douyin.com{suffix}",
    scheme=VALID_SCHEMES,
    host=NON_DOUYIN_DOMAINS,
    suffix=st.from_regex(r"[a-zA-Z0-9/]{0,10}", fullmatch=True),
)


# --- Tests ---

@pytest.mark.property
class TestURLValidationProperty:
    """Property 1: URL Validation — Only Douyin URLs Accepted.

    **Validates: Requirements 1.2**
    """

    def setup_method(self):
        self.downloader = VideoDownloader()

    @given(url=valid_douyin_urls)
    @settings(max_examples=100)
    def test_valid_douyin_urls_accepted(self, url: str):
        """Valid Douyin URLs (various subdomains + paths) are accepted."""
        assert self.downloader.validate_url(url) is True

    @given(url=non_douyin_urls)
    @settings(max_examples=100)
    def test_non_douyin_urls_rejected(self, url: str):
        """Non-Douyin URLs (random domains) are rejected."""
        assert self.downloader.validate_url(url) is False

    @given(url=malformed_strings)
    @settings(max_examples=100)
    def test_malformed_strings_rejected(self, url: str):
        """Malformed strings that are not valid URLs are rejected."""
        # A malformed string should not be accepted unless it happens to be
        # a valid URL with douyin.com domain
        result = self.downloader.validate_url(url)
        # If it's accepted, verify it actually is a douyin.com URL
        if result is True:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            hostname = parsed.hostname.lower() if parsed.hostname else ""
            assert hostname == "douyin.com" or hostname.endswith(".douyin.com")

    @given(url=douyin_in_path_urls)
    @settings(max_examples=100)
    def test_douyin_in_path_not_accepted(self, url: str):
        """URLs with 'douyin.com' in the path/query but not in the hostname are rejected."""
        assert self.downloader.validate_url(url) is False

    @given(url=tricky_domain_urls)
    @settings(max_examples=100)
    def test_tricky_domains_rejected(self, url: str):
        """Tricky domains like 'notdouyin.com', 'douyin.com.evil.com' are rejected."""
        assert self.downloader.validate_url(url) is False
