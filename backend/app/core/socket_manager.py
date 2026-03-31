import socketio
from loguru import logger
from typing import Any, Optional

class SocketManager:
    """
    Manages WebSocket connections using Socket.IO.
    Allows broadcasting events from anywhere in the application.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SocketManager, cls).__new__(cls)
            cls._instance.sio = socketio.AsyncServer(
                async_mode='asgi',
                cors_allowed_origins='*'
            )
            cls._instance.app = socketio.ASGIApp(cls._instance.sio)
            cls._instance._setup_handlers()
        return cls._instance
    
    def _setup_handlers(self):
        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"Client connected: {sid}")
            await self.sio.emit('status', {'connected': True}, to=sid)
            
        @self.sio.event
        async def disconnect(sid):
            logger.info(f"Client disconnected: {sid}")
            
    async def broadcast(self, event: str, data: Any = None):
        """Broadcast an event to all connected clients."""
        logger.debug(f"Broadcasting event: {event}")
        await self.sio.emit(event, data)

# Global instances
socket_manager = SocketManager()
sio_app = socket_manager.app
sio = socket_manager.sio
