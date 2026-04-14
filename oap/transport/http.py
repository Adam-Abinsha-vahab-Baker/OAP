from __future__ import annotations

import asyncio

import httpx

from oap.envelope import TaskEnvelope

_RETRY_DELAYS = [1.0, 2.0, 4.0]  # seconds between attempts 1→2, 2→3, 3→4


class HTTPTransport:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def get(self, path: str = "/") -> httpx.Response:
        """Issue a GET request. Raises httpx exceptions on failure."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.get(f"{self.base_url}{path}")

    async def invoke(self, envelope: TaskEnvelope) -> dict:
        payload = envelope.model_dump(mode="json")
        last_exc: Exception | None = None
        max_attempts = len(_RETRY_DELAYS) + 1  # 4 total

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                await asyncio.sleep(_RETRY_DELAYS[attempt - 2])

            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/invoke",
                        json=payload,
                    )

                if 400 <= response.status_code < 500:
                    # 4xx — fail immediately, do not retry
                    raise _make_error(
                        self.base_url, attempt,
                        f"HTTP {response.status_code} (not retried)",
                        retried=False,
                    )

                if response.status_code >= 500:
                    last_exc = _make_error(
                        self.base_url, attempt,
                        f"HTTP {response.status_code}",
                    )
                    continue

                return response.json()

            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_exc = exc
                continue

        raise _make_error(
            self.base_url,
            max_attempts,
            str(last_exc),
            retried=True,
        )


def _make_error(
    base_url: str,
    attempts: int,
    detail: str,
    retried: bool = True,
) -> Exception:
    from oap.router import RoutingError

    suffix = f" after {attempts} attempt(s)" if retried else ""
    return RoutingError(f"Transport error for {base_url}{suffix}: {detail}")
