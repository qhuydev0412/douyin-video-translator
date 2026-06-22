"""Middleware package for the Douyin Video Translator."""

from app.middleware.bot_protection import BotProtectionMiddleware

__all__ = ["BotProtectionMiddleware"]
