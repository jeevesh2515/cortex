"""HTTP routers, one per surface area.

Grouped so a contributor adding, say, a new ``/api/recipes`` endpoint knows
exactly where it goes without reading every router file. Each module owns its
own schema variations and middleware (CORS is applied globally in app.py).
"""

from cortex.server.routers import chat, memory, search, system

__all__ = ["chat", "memory", "search", "system"]
