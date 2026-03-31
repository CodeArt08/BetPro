import { io, Socket } from 'socket.io-client';

class SocketService {
  private socket: Socket | null = null;
  private subscribers: Map<string, Set<Function>> = new Map();

  connect() {
    if (this.socket?.connected) return;

    // In development, the backend is usually on a different port (8000)
    // In production, we assume they are served from the same origin
    const url = window.location.hostname === 'localhost' 
      ? 'http://localhost:8000' 
      : window.location.origin;

    this.socket = io(url, {
      path: '/socket.io',
      transports: ['websocket', 'polling']
    });

    this.socket.on('connect', () => {
      console.log('Connected to real-time updates server');
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from real-time updates server');
    });

    // Handle generic event broadcasting
    this.socket.onAny((eventName, data) => {
      console.log(`Real-time event: ${eventName}`, data);
      const eventSubscribers = this.subscribers.get(eventName);
      if (eventSubscribers) {
        eventSubscribers.forEach(callback => callback(data));
      }
      
      // Also notify 'any' subscribers
      const anySubscribers = this.subscribers.get('any');
      if (anySubscribers) {
        anySubscribers.forEach(callback => callback(eventName, data));
      }
    });
  }

  subscribe(event: string, callback: Function) {
    if (!this.subscribers.has(event)) {
      this.subscribers.set(event, new Set());
    }
    this.subscribers.get(event)?.add(callback);

    return () => {
      this.subscribers.get(event)?.delete(callback);
    };
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

export const socketService = new SocketService();
export default socketService;
