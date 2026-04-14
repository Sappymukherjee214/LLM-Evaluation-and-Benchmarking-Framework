import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line
} from 'recharts';
import { 
  Activity, Database, Cpu, Settings, Play, History, 
  MessageSquare, ChevronRight, CheckCircle, AlertCircle, Loader2
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const App = () => {
  const [activeTab, setActiveTab] = useState('evaluation');
  const [models, setModels] = useState({ providers: [], popular: [] });
  const [datasets, setDatasets] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState(false);

  useEffect(() => {
    fetchMetadata();
    fetchExperiments();
  }, []);

  const fetchMetadata = async () => {
    try {
      const modRes = await axios.get(`${API_BASE}/models`);
      const dsRes = await axios.get(`${API_BASE}/datasets`);
      setModels(modRes.data);
      setDatasets(dsRes.data);
      setIsBackendConnected(true);
    } catch (err) {
      console.error("Failed to fetch metadata", err);
      setIsBackendConnected(false);
    }
  };

  const fetchExperiments = async () => {
    try {
      const res = await axios.get(`${API_BASE}/experiments`);
      setExperiments(res.data);
    } catch (err) {
      console.error("Failed to fetch experiments", err);
    }
  };

  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 p-6 flex flex-col gap-8">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-black rounded-lg flex items-center justify-center">
            <Activity className="text-white w-5 h-5" />
          </div>
          <h1 className="font-bold text-xl tracking-tight">LLM Eval</h1>
        </div>

        <nav className="flex flex-col gap-2">
          <NavItem 
            icon={<Play size={20} />} 
            label="Evaluation" 
            active={activeTab === 'evaluation'} 
            onClick={() => setActiveTab('evaluation')}
          />
          <NavItem 
            icon={<MessageSquare size={20} />} 
            label="Prompt Tester" 
            active={activeTab === 'prompt'} 
            onClick={() => setActiveTab('prompt')}
          />
          <NavItem 
            icon={<History size={20} />} 
            label="History" 
            active={activeTab === 'history'} 
            onClick={() => setActiveTab('history')}
          />
        </nav>

        <div className="mt-auto p-4 bg-gray-50 rounded-lg border border-gray-100">
           <div className="flex items-center gap-2">
             <div className={`w-2 h-2 rounded-full ${isBackendConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
             <span className="text-xs font-semibold text-gray-500">
               {isBackendConnected ? 'Backend Connected' : 'Backend Disconnected'}
             </span>
           </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-10 overflow-auto">
        {activeTab === 'evaluation' && (
          <EvaluationPanel 
            models={models} 
            datasets={datasets} 
            onRun={() => fetchExperiments()} 
          />
        )}
        {activeTab === 'prompt' && <PromptTester models={models} />}
        {activeTab === 'history' && <HistoryPanel experiments={experiments} />}
      </main>
    </div>
  );
};

const NavItem = ({ icon, label, active, onClick }) => (
  <button 
    onClick={onClick}
    className={`flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-all ${
      active ? 'bg-black text-white shadow-lg' : 'text-gray-500 hover:bg-gray-100'
    }`}
  >
    {icon}
    <span>{label}</span>
  </button>
);

const EvaluationPanel = ({ models, datasets, onRun }) => {
  const [form, setForm] = useState({
    provider: 'mock',
    model: 'mock-model-a',
    dataset: '',
    task: 'qa',
    template: ''
  });
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);

  const handleRun = async () => {
    setRunning(true);
    setResult(null);
    try {
      const res = await axios.post(`${API_BASE}/evaluate`, {
        model_provider: form.provider,
        model_id: form.model,
        dataset_name: form.dataset.replace('.json', ''),
        dataset_path: form.dataset,
        task_type: form.task,
        prompt_template: form.template || null
      });
      setResult(res.data);
      onRun();
    } catch (err) {
      alert("Evaluation failed: " + err.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col gap-8 max-w-5xl">
      <header>
        <h2 className="text-3xl font-bold">New Evaluation</h2>
        <p className="text-gray-500 mt-2">Set up a benchmarking run across your models and datasets.</p>
      </header>

      <div className="grid grid-cols-3 gap-8">
        <div className="col-span-1 card flex flex-col gap-4">
          <label className="text-sm font-semibold text-gray-700">Model Provider</label>
          <select 
            className="input-field" 
            value={form.provider}
            onChange={(e) => setForm({...form, provider: e.target.value})}
          >
            {models.providers.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
          </select>

          <label className="text-sm font-semibold text-gray-700">Model ID</label>
          <input 
            className="input-field" 
            value={form.model}
            onChange={(e) => setForm({...form, model: e.target.value})}
          />

          <label className="text-sm font-semibold text-gray-700">Dataset</label>
          <select 
            className="input-field"
            value={form.dataset}
            onChange={(e) => setForm({...form, dataset: e.target.value})}
          >
            <option value="">Select Dataset</option>
            {datasets.map(d => <option key={d} value={d}>{d}</option>)}
          </select>

          <label className="text-sm font-semibold text-gray-700">Task Type</label>
          <select 
            className="input-field"
            value={form.task}
            onChange={(e) => setForm({...form, task: e.target.value})}
          >
            <option value="qa">QA</option>
            <option value="summarization">Summarization</option>
            <option value="classification">Classification</option>
          </select>

          <button 
            className="btn-primary mt-4 flex items-center justify-center gap-2"
            disabled={running || !form.dataset}
            onClick={handleRun}
          >
            {running ? <Loader2 className="animate-spin" size={20} /> : <Play size={18} />}
            {running ? 'Running...' : 'Start Evaluation'}
          </button>
        </div>

        <div className="col-span-2">
           {result ? (
             <div className="card h-full">
               <div className="flex justify-between items-center mb-6">
                 <h3 className="font-bold text-lg">Results: {result.config.experiment_id}</h3>
                 <span className="bg-green-100 text-green-700 px-3 py-1 rounded-full text-sm font-medium">Completed</span>
               </div>

               <div className="grid grid-cols-2 gap-4 mb-8">
                  <StatCard label="Accuracy" value={`${(result.summary_metrics.accuracy * 100).toFixed(1)}%`} />
                  <StatCard label="Avg Latency" value={`${result.summary_metrics.latency.toFixed(2)}s`} />
               </div>

               <div className="h-64 mt-4">
                 <ResponsiveContainer width="100%" height="100%">
                   <BarChart data={result.results.slice(0, 5)}>
                     <CartesianGrid strokeDasharray="3 3" vertical={false} />
                     <XAxis dataKey="sample_id" />
                     <YAxis />
                     <Tooltip />
                     <Bar dataKey="metrics.accuracy" fill="#000" radius={[4, 4, 0, 0]} />
                   </BarChart>
                 </ResponsiveContainer>
               </div>
             </div>
           ) : (
             <div className="h-full border-2 border-dashed border-gray-200 rounded-xl flex flex-col items-center justify-center text-gray-400 gap-4">
               <Cpu size={48} strokeWidth={1} />
               <p>Select parameters and run to see results</p>
             </div>
           )}
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ label, value }) => (
  <div className="bg-gray-50 p-4 rounded-xl border border-gray-100">
    <p className="text-xs font-bold text-gray-400 uppercase tracking-widest">{label}</p>
    <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
  </div>
);

const PromptTester = ({ models }) => {
  const [prompt, setPrompt] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleTest = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/test-prompt`, {
        prompt,
        models: [
          { provider: 'mock', id: 'GPT-4-Mock' },
          { provider: 'mock', id: 'Llama-3-Mock' }
        ]
      });
      setResults(res.data);
    } catch (err) {
      alert("Prompt test failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-8 max-w-5xl">
       <header>
        <h2 className="text-3xl font-bold">Prompt Tester</h2>
        <p className="text-gray-500 mt-2">Test your prompts side-by-side across multiple models.</p>
      </header>

      <div className="card flex flex-col gap-4">
        <textarea 
          className="input-field min-h-[150px] font-mono text-sm"
          placeholder="Enter your test prompt here..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <div className="flex justify-end">
          <button 
            className="btn-primary flex items-center gap-2"
            disabled={loading || !prompt}
            onClick={handleTest}
          >
            {loading ? <Loader2 className="animate-spin" size={18} /> : <Play size={18} />}
            Run Comparison
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {results.map((res, i) => (
          <div key={i} className="card">
            <div className="flex justify-between items-center mb-4">
              <h4 className="font-bold text-gray-700">{res.model}</h4>
              <span className="text-xs text-gray-400 font-mono">{res.latency.toFixed(2)}s</span>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg text-sm whitespace-pre-wrap min-h-[100px] border border-gray-100">
              {res.output}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const HistoryPanel = ({ experiments }) => (
  <div className="flex flex-col gap-8 max-w-6xl">
    <header>
      <h2 className="text-3xl font-bold">Experiment History</h2>
      <p className="text-gray-500 mt-2">Overview of all past benchmarking runs.</p>
    </header>

    <div className="card p-0 overflow-hidden">
      <table className="w-full text-left border-collapse">
        <thead className="bg-gray-50 border-b border-gray-100">
          <tr>
            <th className="px-6 py-4 text-xs font-bold text-gray-400 uppercase">Experiment ID</th>
            <th className="px-6 py-4 text-xs font-bold text-gray-400 uppercase">Model</th>
            <th className="px-6 py-4 text-xs font-bold text-gray-400 uppercase">Dataset</th>
            <th className="px-6 py-4 text-xs font-bold text-gray-400 uppercase text-center">Accuracy</th>
            <th className="px-6 py-4 text-xs font-bold text-gray-400 uppercase text-center">Avg Latency</th>
            <th className="px-6 py-4 text-xs font-bold text-gray-400 uppercase">Timestamp</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {experiments.map((exp) => (
            <tr key={exp.id} className="hover:bg-gray-50 transition-colors">
              <td className="px-6 py-4 font-mono text-sm">{exp.id}</td>
              <td className="px-6 py-4 font-medium">{exp.model}</td>
              <td className="px-6 py-4 text-gray-500">{exp.dataset}</td>
              <td className="px-6 py-4 text-center">
                <span className={`px-2 py-1 rounded font-bold text-xs ${
                  exp.accuracy > 0.8 ? 'bg-green-100 text-green-700' : 
                  exp.accuracy > 0.5 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'
                }`}>
                  {(exp.accuracy * 100).toFixed(1)}%
                </span>
              </td>
              <td className="px-6 py-4 text-center text-gray-600 font-mono text-sm">{exp.latency.toFixed(3)}s</td>
              <td className="px-6 py-4 text-xs text-gray-400">{new Date(exp.timestamp).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

export default App;
