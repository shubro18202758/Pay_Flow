"""PayFlow — API Route Registration."""

from src.api.routes.analyst import router as analyst_router
from src.api.routes.dashboard import router as dashboard_router
from src.api.routes.fraud import router as fraud_router

__all__ = ["analyst_router", "dashboard_router", "fraud_router"]
