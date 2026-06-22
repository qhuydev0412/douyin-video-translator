"""Middleware to block common vulnerability scanners and bots.

Drops connections from requests targeting well-known scan paths,
suspicious file extensions, and common fingerprinting endpoints.
"""

import logging
import time
from collections import defaultdict
from typing import Callable

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths that scanners commonly probe — immediate block
BLOCKED_PATHS: set[str] = {
    # Environment / config leaks
    "/.env",
    "/.env.local",
    "/.env.production",
    "/config.json",
    "/config.yml",
    "/.git/config",
    "/.git/HEAD",
    # Static assets that don't exist
    "/favicon.ico",
    # JS framework fingerprinting
    "/_app/version.json",
    "/_payload.json",
    "/__manifest",
    "/assets/index.js",
    "/main.js.map",
    "/bundle.js.map",
    "/app.js.map",
    # Firebase
    "/__/firebase/init.json",
    "/__/firebase/init.js",
    # Cloudflare
    "/cdn-cgi/trace",
    # GraphQL / Admin panels
    "/graphql",
    "/admin",
    "/login",
    "/wp-admin",
    "/wp-login.php",
    # API docs (block if you don't need public docs)
    "/swagger.json",
    "/api-docs",
    "/reference",
    # AI service probing
    "/services/openai.ts",
    "/services/anthropic.ts",
    "/services/openai.js",
    "/services/anthropic.js",
}

# File extensions that are never served by this API
BLOCKED_EXTENSIONS: tuple[str, ...] = (
    ".map",
    ".php",
    ".asp",
    ".aspx",
    ".jsp",
    ".cgi",
    ".xml",
    ".sql",
    ".bak",
    ".old",
    ".DS_Store",
)

# Suspicious user-agent substrings (case-insensitive matching)
BLOCKED_USER_AGENTS: tuple[str, ...] = (
    "sqlmap",
    "nikto",
    "nmap",
    "masscan",
    "zgrab",
    "gobuster",
    "dirbuster",
    "nuclei",
    "httpx",
    "censys",
    "shodan",
)


class BotProtectionMiddleware(BaseHTTPMiddleware):
    """Drop requests from known vulnerability scanners.

    Features:
    - Block known scan paths (immediate 444 — connection drop)
    - Block suspicious file extensions
    - Block known scanner user-agents
    - Simple per-IP 404 rate tracking (block IPs with too many 404s)
    """

    def __init__(
        self,
        app: ASGIApp,
        max_404_per_minute: int = 15,
        block_duration_seconds: int = 300,
    ) -> None:
        super().__init__(app)
        self.max_404_per_minute = max_404_per_minute
        self.block_duration_seconds = block_duration_seconds
        # Track 404 counts: {ip: [(timestamp, count)]}
        self._404_counts: dict[str, list[float]] = defaultdict(list)
        self._blocked_ips: dict[str, float] = {}  # {ip: blocked_until_timestamp}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = self._get_client_ip(request)
        path = request.url.path.rstrip("/") or "/"

        # 1. Check if IP is temporarily blocked
        if self._is_ip_blocked(client_ip):
            return Response(status_code=403)

        # 2. Check blocked paths
        if path in BLOCKED_PATHS or path.lower() in BLOCKED_PATHS:
            logger.debug("Blocked scan path: %s from %s", path, client_ip)
            self._record_suspicious(client_ip)
            return Response(status_code=444)

        # 3. Check blocked extensions
        if path.endswith(BLOCKED_EXTENSIONS):
            logger.debug("Blocked extension: %s from %s", path, client_ip)
            self._record_suspicious(client_ip)
            return Response(status_code=444)

        # 4. Check user-agent
        ua = (request.headers.get("user-agent") or "").lower()
        if any(scanner in ua for scanner in BLOCKED_USER_AGENTS):
            logger.debug("Blocked scanner UA: %s from %s", ua, client_ip)
            self._record_suspicious(client_ip)
            return Response(status_code=444)

        # 5. Process request normally
        response = await call_next(request)

        # 6. Track 404s for rate limiting
        if response.status_code == 404:
            self._record_404(client_ip)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For behind proxy/ingress."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP is currently blocked."""
        if ip in self._blocked_ips:
            if time.time() < self._blocked_ips[ip]:
                return True
            else:
                del self._blocked_ips[ip]
        return False

    def _record_404(self, ip: str) -> None:
        """Record a 404 for this IP and block if threshold exceeded."""
        now = time.time()
        window = now - 60  # 1-minute window

        # Clean old entries
        self._404_counts[ip] = [t for t in self._404_counts[ip] if t > window]
        self._404_counts[ip].append(now)

        if len(self._404_counts[ip]) > self.max_404_per_minute:
            self._blocked_ips[ip] = now + self.block_duration_seconds
            logger.warning(
                "Blocked IP %s for %ds — exceeded %d 404s/min",
                ip,
                self.block_duration_seconds,
                self.max_404_per_minute,
            )

    def _record_suspicious(self, ip: str) -> None:
        """Record multiple 404-equivalents for a blocked path hit."""
        # Count each blocked path hit as 3 strikes
        for _ in range(3):
            self._record_404(ip)
