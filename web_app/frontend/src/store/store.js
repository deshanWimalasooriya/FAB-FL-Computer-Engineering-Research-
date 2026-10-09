import { create } from 'zustand';

// Store for managing FAB-FL Server Dashboard State
const useStore = create((set, get) => ({
  // --- State Variables ---
  isConnected: false,
  isTraining: false,
  trainingProgress: 0,
  currentRound: 0,
  globalAccuracy: 0,
  globalLoss: 0,
  nodes: [],
  socket: null,

  // --- Actions ---
  
  // Establish WebSocket Connection
  connectWebSocket: () => {
    // Prevent multiple connections
    if (get().socket) return;

    const ws = new WebSocket('ws://localhost:8000/ws/metrics');

    ws.onopen = () => {
      console.log('WebSocket Connected');
      set({ isConnected: true, socket: ws });
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'metrics') {
        // Update dashboard state with incoming real-time data
        set({
          currentRound: data.round,
          trainingProgress: data.progress,
          nodes: data.nodes,
          globalAccuracy: data.globalAccuracy || data.global_accuracy,
          globalLoss: data.globalLoss || data.global_loss,
        });
      } else if (data.type === 'status') {
        if (data.message === 'Training Complete') {
          set({ isTraining: false });
        }
      }
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      set({ isConnected: false, socket: null });
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        get().connectWebSocket();
      }, 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error: ', error);
      ws.close();
    };
  },

  // Disconnect WebSocket
  disconnectWebSocket: () => {
    const { socket } = get();
    if (socket) {
      socket.close();
    }
  },

  // API Call to Start Training
  startTraining: async () => {
    try {
      const res = await fetch('http://localhost:8000/api/start', { method: 'POST' });
      if (res.ok) {
        set({ isTraining: true, currentRound: 0, trainingProgress: 0 });
      }
    } catch (error) {
      console.error('Failed to start training:', error);
    }
  },

  // API Call to Stop Training
  stopTraining: async () => {
    try {
      const res = await fetch('http://localhost:8000/api/stop', { method: 'POST' });
      if (res.ok) {
        set({ isTraining: false });
      }
    } catch (error) {
      console.error('Failed to stop training:', error);
    }
  },

  // API Call to Restart Training
  restartTraining: async () => {
    try {
      const res = await fetch('http://localhost:8000/api/restart', { method: 'POST' });
      if (res.ok) {
        set({ isTraining: true, currentRound: 0, trainingProgress: 0 });
      }
    } catch (error) {
      console.error('Failed to restart training:', error);
    }
  },
}));

export default useStore;
