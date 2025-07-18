from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from time import time

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit: int = 100, time_window: int = 60):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.clients = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time()

        if client_ip not in self.clients:
            self.clients[client_ip] = []

        # Remove timestamps that are outside the time window
        self.clients[client_ip] = [
            timestamp for timestamp in self.clients[client_ip]
            if current_time - timestamp < self.time_window
        ]

        if len(self.clients[client_ip]) >= self.rate_limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Record the current request timestamp
        self.clients[client_ip].append(current_time)

        response = await call_next(request)
        return response