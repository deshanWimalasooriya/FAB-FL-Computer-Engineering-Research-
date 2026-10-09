import React, { useEffect, useState } from 'react';
import useStore from '../store/store';
import { 
  Play, Square, RotateCcw, Activity, 
  Server, Settings, Cpu, HardDrive, 
  Home, Users, PieChart, Terminal
} from 'lucide-react';

const SidebarIcon = ({ icon: Icon, label, active }) => (
  <div className={`flex flex-col items-center justify-center w-16 h-16 mt-4 mb-4 rounded-xl cursor-pointer transition-all duration-300 ease-in-out hover:bg-slate-700/50 hover:shadow-[0_0_15px_rgba(0,229,255,0.4)] ${active ? 'text-cyan-400 bg-slate-800 border-l-2 border-cyan-400' : 'text-slate-400'}`}>
    <Icon size={24} />
    <span className="text-[10px] mt-1 font-medium">{label}</span>
  </div>
);

const Dashboard = () => {
  const { 
    isConnected, 
    isTraining, 
    trainingProgress, 
    currentRound, 
    nodes,
    globalAccuracy,
    globalLoss,
    connectWebSocket, 
    disconnectWebSocket,
    startTraining,
    stopTraining,
    restartTraining
  } = useStore();

  const [freqsel, setFreqsel] = useState(true);

  // Initialize WebSocket connection on component mount
  useEffect(() => {
    connectWebSocket();
    return () => disconnectWebSocket();
  }, [connectWebSocket, disconnectWebSocket]);

  return (
    <div className="flex h-screen w-full bg-slate-900 text-slate-100 font-sans overflow-hidden">
      
      {/* Sidebar Navigation */}
      <nav className="w-24 bg-slate-900/80 border-r border-slate-700/50 flex flex-col items-center py-6 shadow-2xl z-10 backdrop-blur-md">
        <div className="mb-8">
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center shadow-[0_0_20px_rgba(0,229,255,0.6)]">
            <Activity className="text-white" size={28} />
          </div>
        </div>
        <div className="flex flex-col flex-1 w-full items-center">
          <SidebarIcon icon={Home} label="Home" active={true} />
          <SidebarIcon icon={PieChart} label="Graph" active={false} />
          <SidebarIcon icon={Users} label="Clients" active={false} />
          <SidebarIcon icon={Server} label="Model" active={false} />
          <SidebarIcon icon={Terminal} label="Terminal" active={false} />
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 p-8 overflow-y-auto bg-slate-950 relative">
        {/* Background ambient glow */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-600/10 rounded-full blur-[120px] pointer-events-none"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-[120px] pointer-events-none"></div>

        {/* Header */}
        <header className="flex justify-between items-center mb-8 relative z-10">
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-white flex items-center gap-3">
              FAB-FL <span className="text-cyan-400">Server Node</span>
            </h1>
            <p className="text-slate-400 mt-1">Federated Learning Orchestration Dashboard</p>
          </div>
          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full backdrop-blur-md border ${isConnected ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-rose-500/10 border-rose-500/30 text-rose-400'}`}>
              <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`}></div>
              <span className="text-sm font-semibold">{isConnected ? 'System Online' : 'System Offline'}</span>
            </div>
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full backdrop-blur-md border ${isTraining ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400' : 'bg-slate-800 border-slate-700 text-slate-400'}`}>
              <Activity className={isTraining ? 'animate-spin' : ''} size={16} />
              <span className="text-sm font-semibold">{isTraining ? 'Training Active' : 'Idle'}</span>
            </div>
          </div>
        </header>

        {/* CSS Grid Layout */}
        <div className="grid grid-cols-12 gap-6 relative z-10">
          
          {/* Server Controls Card */}
          <div className="col-span-12 md:col-span-4 bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl transition-all duration-300 hover:border-cyan-500/30 hover:shadow-[0_0_30px_rgba(0,229,255,0.05)]">
            <h2 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
              <Server className="text-cyan-400" size={20} />
              Server Controls
            </h2>
            <div className="flex flex-col gap-4">
              <button 
                onClick={startTraining}
                disabled={isTraining}
                className="group relative flex items-center justify-center gap-2 w-full py-4 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 text-white font-bold text-lg transition-all hover:scale-[1.02] active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed overflow-hidden"
              >
                <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-in-out"></div>
                <Play fill="currentColor" size={20} className="relative z-10" />
                <span className="relative z-10">START TRAINING</span>
              </button>
              
              <div className="grid grid-cols-2 gap-4">
                <button 
                  onClick={stopTraining}
                  disabled={!isTraining}
                  className="flex items-center justify-center gap-2 py-3 rounded-xl bg-slate-800/80 border border-slate-600 text-slate-300 hover:text-rose-400 hover:border-rose-400 hover:bg-rose-950/30 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Square fill="currentColor" size={16} />
                  STOP
                </button>
                <button 
                  onClick={restartTraining}
                  className="flex items-center justify-center gap-2 py-3 rounded-xl bg-slate-800/80 border border-slate-600 text-slate-300 hover:text-cyan-400 hover:border-cyan-400 hover:bg-cyan-950/30 transition-all active:scale-95"
                >
                  <RotateCcw size={16} />
                  RESTART
                </button>
              </div>
            </div>

            <div className="mt-8 pt-6 border-t border-slate-700/50">
               <div className="flex justify-between items-center mb-2">
                 <span className="text-sm text-slate-400">Global Accuracy</span>
                 <span className="text-emerald-400 font-mono font-bold">{(globalAccuracy * 100).toFixed(2)}%</span>
               </div>
               <div className="flex justify-between items-center">
                 <span className="text-sm text-slate-400">Global Loss</span>
                 <span className="text-rose-400 font-mono font-bold">{globalLoss?.toFixed(4) || "0.0000"}</span>
               </div>
            </div>
          </div>

          {/* Training Controls Card */}
          <div className="col-span-12 md:col-span-8 bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl transition-all duration-300 hover:border-cyan-500/30 hover:shadow-[0_0_30px_rgba(0,229,255,0.05)]">
            <h2 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
              <Settings className="text-cyan-400" size={20} />
              Training Configuration
            </h2>
            <div className="grid grid-cols-3 gap-6">
              <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700/50">
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Total Rounds</label>
                <input type="number" defaultValue={100} className="w-full bg-transparent text-white text-2xl font-mono focus:outline-none focus:text-cyan-400 transition-colors" />
              </div>
              <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700/50">
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">M (Clients per round)</label>
                <input type="number" defaultValue={10} className="w-full bg-transparent text-white text-2xl font-mono focus:outline-none focus:text-cyan-400 transition-colors" />
              </div>
              <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700/50">
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">K (Total Clients)</label>
                <input type="number" defaultValue={100} className="w-full bg-transparent text-white text-2xl font-mono focus:outline-none focus:text-cyan-400 transition-colors" />
              </div>
            </div>

            <div className="mt-8 flex items-center justify-between p-4 bg-slate-900/50 rounded-xl border border-slate-700/50">
              <div>
                <h3 className="text-white font-medium">Enable FREQSEL Filter</h3>
                <p className="text-sm text-slate-400">Frequency-based dynamic client selection</p>
              </div>
              {/* Custom sleek toggle switch */}
              <button 
                onClick={() => setFreqsel(!freqsel)}
                className={`relative w-14 h-8 rounded-full transition-colors duration-300 ease-in-out focus:outline-none ${freqsel ? 'bg-cyan-500' : 'bg-slate-600'}`}
              >
                <div className={`absolute top-1 w-6 h-6 rounded-full bg-white transition-transform duration-300 ease-in-out ${freqsel ? 'left-7 shadow-[0_0_10px_rgba(255,255,255,0.8)]' : 'left-1'}`}></div>
              </button>
            </div>
          </div>

          {/* UCB Joint Scoring Metrics Terminal */}
          <div className="col-span-12 md:col-span-5 bg-slate-900/80 backdrop-blur-xl border border-slate-700/80 rounded-2xl p-1 shadow-2xl overflow-hidden font-mono text-sm relative group">
            <div className="absolute inset-0 border border-cyan-500/0 group-hover:border-cyan-500/30 rounded-2xl transition-colors duration-500 pointer-events-none"></div>
            <div className="bg-slate-950 p-4 rounded-xl h-full border border-slate-800">
              <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-2">
                <h2 className="text-cyan-400 font-bold flex items-center gap-2">
                  <Terminal size={16} />
                  UCB_JOINT_SCORING_LOG
                </h2>
                <div className="flex gap-2">
                  <div className="w-3 h-3 rounded-full bg-rose-500/80"></div>
                  <div className="w-3 h-3 rounded-full bg-amber-500/80"></div>
                  <div className="w-3 h-3 rounded-full bg-emerald-500/80"></div>
                </div>
              </div>
              
              <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                {nodes.length > 0 ? nodes.map((node, i) => (
                  <div key={i} className="text-slate-300 bg-slate-900 p-2 rounded border border-slate-800/50">
                    <div className="text-cyan-300 mb-1">&gt; Analyzing {node.id} ...</div>
                    <div className="grid grid-cols-2 gap-2 pl-4">
                      <span className="text-slate-500">Skew:</span> 
                      <span className="text-amber-300">{node.data_skew}</span>
                      
                      <span className="text-slate-500">Latency:</span> 
                      <span className={node.status === 'Online' ? 'text-emerald-300' : 'text-slate-600'}>{node.latency}</span>
                      
                      <span className="text-slate-500">UCB Score:</span> 
                      <span className="text-blue-400 font-bold">{node.ucb_score}</span>
                    </div>
                  </div>
                )) : (
                  <div className="text-slate-500 italic">&gt; Waiting for metrics stream...</div>
                )}
                {/* Simulated typing cursor */}
                {isTraining && <div className="w-2 h-4 bg-cyan-400 animate-pulse mt-2"></div>}
              </div>
            </div>
          </div>

          {/* Connected Edge Nodes Table */}
          <div className="col-span-12 md:col-span-7 bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl transition-all duration-300 hover:border-cyan-500/30 hover:shadow-[0_0_30px_rgba(0,229,255,0.05)]">
            <h2 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
              <HardDrive className="text-cyan-400" size={20} />
              Connected Edge Nodes
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-700 text-slate-400 text-sm">
                    <th className="pb-3 px-2 font-medium">Node ID</th>
                    <th className="pb-3 px-2 font-medium">IP Address</th>
                    <th className="pb-3 px-2 font-medium">Hardware</th>
                    <th className="pb-3 px-2 font-medium">Clients</th>
                    <th className="pb-3 px-2 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {nodes.length > 0 ? nodes.map((node, idx) => (
                    <tr key={idx} className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors group">
                      <td className="py-4 px-2 text-white font-medium flex items-center gap-2">
                        <Cpu size={16} className="text-cyan-400/70 group-hover:text-cyan-400 transition-colors" />
                        {node.id}
                      </td>
                      <td className="py-4 px-2 text-slate-300 font-mono text-sm">{node.ip}</td>
                      <td className="py-4 px-2 text-slate-300">{node.hardware}</td>
                      <td className="py-4 px-2 text-slate-300">{node.clients}</td>
                      <td className="py-4 px-2">
                        <span className={`px-2 py-1 rounded text-xs font-semibold ${node.status === 'Online' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-slate-700/50 text-slate-400 border border-slate-600'}`}>
                          {node.status}
                        </span>
                      </td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan="5" className="py-8 text-center text-slate-500">No nodes connected. Waiting for network...</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

        </div>

        {/* Global Progress Bar */}
        <div className="fixed bottom-0 left-24 right-0 h-1.5 bg-slate-800 z-50">
          <div 
            className="h-full bg-cyan-400 shadow-[0_0_15px_rgba(0,229,255,0.8)] transition-all duration-1000 ease-in-out relative"
            style={{ width: `${trainingProgress}%` }}
          >
            {isTraining && (
              <div className="absolute top-[-30px] right-0 translate-x-1/2 bg-slate-800 text-cyan-400 px-2 py-1 rounded text-xs font-bold border border-cyan-500/30">
                Round {currentRound}
              </div>
            )}
          </div>
        </div>
        
        {/* Custom scrollbar styles inject */}
        <style dangerouslySetInnerHTML={{__html: `
          .custom-scrollbar::-webkit-scrollbar { width: 4px; }
          .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
          .custom-scrollbar::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
          .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #00E5FF; }
        `}} />
      </main>
    </div>
  );
};

export default Dashboard;
