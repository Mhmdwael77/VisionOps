import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Camera,
  Download,
  Gauge,
  LineChart,
  Play,
  Settings,
  Upload,
  Users,
  Video,
  Waypoints
} from "lucide-react";
import { PanelLeftClose, PanelLeftOpen, ScanLine, GitBranch } from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import "./styles.css";
import "./review.css";
import "./theme.css";
import { RetailTest } from "./RetailTest";
import { OverviewDashboard } from "./Overview";
import { TrackingSettings } from "./TrackingSettings";
import { ReviewProvider, ReviewWorkspace } from "./ReviewWorkspace";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type EventRow = {
  id: number;
  job_id: string;
  track_id: number;
  event_type: string;
  timestamp: number;
  label: string | null;
};

type Job = {
  id: string;
  source_path: string;
  output_path: string | null;
  status: string;
  progress: number;
  error: string | null;
  entries: number;
  exits: number;
  current_occupancy: number;
};

type Config = {
  model_path: string;
  model_available: boolean;
  fallback_model: string;
  detector_confidence: number;
  detector_iou: number;
  detector_imgsz: number;
  frame_interval: number;
  counting_lines: Array<{ name: string; x1: number; y1: number; x2: number; y2: number; entry_direction: string; min_displacement: number }>;
  zones: Array<{ name: string; polygon: Array<{ x: number; y: number }> }>;
};

type ZonePoint = { zone_name: string; tracker_id_count: number; visits: number; average_dwell_seconds: number | null };
type CameraRow = { id: number; name: string; source: string; source_type: string; status: string };

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function StatCard({ icon, label, value, sub }: { icon: React.ReactNode; label: string; value: string | number; sub: string }) {
  return (
    <section className="stat-card">
      <div className="icon-box">{icon}</div>
      <div>
        <p className="label">{label}</p>
        <strong>{value}</strong>
        <span>{sub}</span>
      </div>
    </section>
  );
}

function App() {
  const [tab, setTab] = useState("overview");
  const [collapsed, setCollapsed] = useState(false);
  const [zones, setZones] = useState<ZonePoint[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [config, setConfig] = useState<Config | null>(null);
  const [cameras, setCameras] = useState<CameraRow[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [sourcePath, setSourcePath] = useState("");
  const [cameraName, setCameraName] = useState("");
  const [cameraSource, setCameraSource] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  async function refresh() {
    const [zoneData, jobData, configData, cameraData] = await Promise.all([
      getJson<ZonePoint[]>("/api/analytics/zones"),
      getJson<Job[]>("/api/videos"),
      getJson<Config>("/api/config"),
      getJson<CameraRow[]>("/api/cameras")
    ]);
    setZones(zoneData);
    setJobs(jobData);
    setConfig(configData);
    setCameras(cameraData);
  }

  useEffect(() => {
    refresh().catch((error) => setMessage(error.message));
    const timer = window.setInterval(() => refresh().catch(() => undefined), 3000);
    return () => window.clearInterval(timer);
  }, []);

  const latestJob = jobs[0];
  const groups = useMemo(
    () => [
      { title: "Insights", items: [
        { id: "overview", Icon: BarChart3, label: "Overview", description: "Store traffic, people tracked and zone activity at a glance." },
        { id: "analytics", Icon: LineChart, label: "Analytics", description: "Traffic trends, occupancy over time and event history." },
        { id: "zones", Icon: Waypoints, label: "Zones", description: "Edit zone geometry and compare engagement and dwell per zone." }
      ] },
      { title: "Review", items: [
        { id: "video-review", Icon: ScanLine, label: "Video Analysis", description: "Frame-accurate playback with detections, zones and corrections." },
        { id: "tracking", Icon: GitBranch, label: "Tracking Inspector", description: "Inspect every tracked identity, then merge or split mistakes." }
      ] },
      { title: "Pipeline", items: [
        { id: "retail-test", Icon: Gauge, label: "Run Analysis", description: "Run the detection and tracking pipeline and tune line and zone geometry." },
        { id: "raw-overview", Icon: Activity, label: "Run History", description: "Every completed analysis with its raw counts and artifacts." },
        { id: "analysis", Icon: Upload, label: "Processing", description: "Upload videos or connect sources and follow processing jobs." },
        { id: "cameras", Icon: Camera, label: "Cameras", description: "Register RTSP, webcam or file sources." },
        { id: "live", Icon: Video, label: "Live", description: "Real-time monitoring.", soon: true }
      ] },
      { title: "System", items: [
        { id: "settings", Icon: Settings, label: "Settings", description: "Detector, frame sampling and tracker configuration." }
      ] }
    ],
    []
  );
  const current = groups.flatMap(group => group.items).find(item => item.id === tab);

  async function uploadAndQueue() {
    if (!file) return;
    setBusy(true);
    setMessage("");
    try {
      const upload = await fetch(`${API}/api/videos/upload`, {
        method: "POST",
        headers: { "x-filename": file.name, "content-type": "application/octet-stream" },
        body: await file.arrayBuffer()
      });
      if (!upload.ok) throw new Error(await upload.text());
      const job = (await upload.json()) as Job;
      await fetch(`${API}/api/jobs/${job.id}/start`, { method: "POST" });
      await refresh();
      setMessage("Upload accepted and processing started.");
    } finally {
      setBusy(false);
    }
  }

  async function startPathJob() {
    if (!sourcePath.trim()) return;
    setBusy(true);
    try {
      const response = await fetch(`${API}/api/jobs`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ source_path: sourcePath.trim() })
      });
      if (!response.ok) throw new Error(await response.text());
      await refresh();
      setMessage("Processing job started.");
    } finally {
      setBusy(false);
    }
  }

  async function saveCamera() {
    if (!cameraName.trim() || !cameraSource.trim()) return;
    await fetch(`${API}/api/cameras`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: cameraName, source: cameraSource, source_type: cameraSource.startsWith("rtsp") ? "rtsp" : "file" })
    });
    setCameraName("");
    setCameraSource("");
    await refresh();
  }

  async function saveSettings() {
    if (!config) return;
    await fetch(`${API}/api/config`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        detector_confidence: config.detector_confidence,
        detector_iou: config.detector_iou,
        detector_imgsz: config.detector_imgsz,
        frame_interval: config.frame_interval
      })
    });
    await refresh();
    setMessage("Processing settings saved for future jobs.");
  }

  function setConfigNumber(key: keyof Pick<Config, "detector_confidence" | "detector_iou" | "detector_imgsz" | "frame_interval">, value: number) {
    setConfig((current) => (current ? { ...current, [key]: value } : current));
  }

  return (
    <main className={`shell ${collapsed ? "collapsed" : ""}`}>
      <aside className="sidebar">
        <div className="brand">
          <Activity size={28} />
          <div>
            <strong>VisionOps</strong>
            <span>Retail video intelligence</span>
          </div>
        </div>
        <button className="sidebar-toggle icon-button" title={collapsed ? "Expand navigation" : "Collapse navigation"} aria-label={collapsed ? "Expand navigation" : "Collapse navigation"} onClick={() => setCollapsed(!collapsed)}>{collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}</button>
        <nav>
          {groups.map(group => (
            <div className="nav-group" key={group.title}>
              <p className="nav-title">{group.title}</p>
              {group.items.map(({ id, Icon, label, ...item }) => (
                <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)} title={label} aria-current={tab === id ? "page" : undefined}>
                  <Icon size={18} />
                  <span>{label}</span>
                  {"soon" in item && <em className="nav-soon">Soon</em>}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className={`model-status ${config?.model_available ? "good" : "warn"}`} title={config?.model_path}>
          <span className="status-dot" />
          <span>{config ? (config.model_available ? "Fine-tuned YOLO11s" : "Fallback model") : "Connecting..."}</span>
        </div>
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">{groups.find(group => group.items.some(item => item.id === tab))?.title}</p>
            <h1>{current?.label ?? tab}</h1>
            {current?.description && <p className="page-description">{current.description}</p>}
          </div>
        </header>

        {message && tab !== "overview" && <div className="notice">{message}</div>}
        {tab === "retail-test" && <RetailTest />}

        {["overview", "video-review", "tracking", "zones", "analytics"].includes(tab) && <ReviewWorkspace view={tab} navigate={setTab} />}
        {tab === "raw-overview" && (
          <OverviewDashboard />
        )}

        {tab === "live" && (
          <section className="empty-state"><Video size={34} /><h2>Live monitoring is coming soon</h2><p>Connect a camera in Cameras. Recorded-video results are available in Overview and Analytics.</p><button className="secondary" onClick={() => setTab("cameras")}><Camera size={16} /> Go to Cameras</button></section>
        )}

        {tab === "analysis" && (
          <section className="grid-2">
            <Panel title="New analysis">
              <label className={`dropzone ${file ? "has-file" : ""}`} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); setFile(event.dataTransfer.files?.[0] ?? null); }}>
                <input type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
                <Upload size={26} />
                <strong>{file ? file.name : "Drop a video here or click to browse"}</strong>
                <span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "MP4, MOV, AVI or MKV"}</span>
              </label>
              <button className="primary wide" disabled={!file || busy} onClick={uploadAndQueue}>
                <Upload size={18} /> Upload and process
              </button>
              <div className="divider"><span>or use a source</span></div>
              <div className="inline-form">
                <input className="input" placeholder="Local path, RTSP URL, or webcam index" value={sourcePath} onChange={(event) => setSourcePath(event.target.value)} />
                <button className="secondary" disabled={!sourcePath || busy} onClick={startPathJob}>
                  <Play size={18} /> Start
                </button>
              </div>
            </Panel>
            <Panel title="Processing Jobs">
              <JobList jobs={jobs} />
            </Panel>
          </section>
        )}

        {tab === "raw-zones" && (
          <section className="grid-2">
            <Panel title="Counting Lines">{config?.counting_lines.length ? config.counting_lines.map((line) => <pre key={line.name}>{JSON.stringify(line, null, 2)}</pre>) : <Empty text="No saved lines. The backend uses a frame-centered default line." />}</Panel>
            <Panel title="Zones">{zones.length ? zones.map((zone) => <Metric key={zone.zone_name} label={zone.zone_name} value={`${zone.tracker_id_count} track IDs / ${zone.visits} visits`} />) : <Empty text="No zone visits yet." />}</Panel>
          </section>
        )}

        {tab === "raw-analytics" && (
          <section className="grid-2">
            <ChartPanel title="Zone Track IDs">
              {zones.length ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={zones}>
                    <CartesianGrid stroke="#213244" />
                    <XAxis dataKey="zone_name" stroke="#8ba2b6" />
                    <YAxis stroke="#8ba2b6" />
                    <Tooltip contentStyle={{ background: "#101922", border: "1px solid #223244" }} />
                    <Bar dataKey="tracker_id_count" name="Track IDs, not unique customers" fill="#18d4ff" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <Empty text="Create zones and process a video to see zone engagement." />
              )}
            </ChartPanel>
            <Panel title="Exports">
              <a className="primary link" href={`${API}/api/exports/events.csv`}>
                <Download size={18} /> Download CSV
              </a>
            </Panel>
          </section>
        )}

        {tab === "cameras" && (
          <section className="grid-2">
            <Panel title="Add Camera">
              <input className="input" placeholder="Camera name" value={cameraName} onChange={(event) => setCameraName(event.target.value)} />
              <input className="input" placeholder="RTSP URL, webcam index, or file path" value={cameraSource} onChange={(event) => setCameraSource(event.target.value)} />
              <button className="primary" onClick={saveCamera}>
                <Camera size={18} /> Save Camera
              </button>
            </Panel>
            <Panel title="Configured Cameras">
              {cameras.length ? cameras.map((camera) => <Metric key={camera.id} label={`${camera.name} (${camera.status})`} value={camera.source} />) : <Empty text="No cameras configured." />}
            </Panel>
          </section>
        )}

        {tab === "settings" && (
          <>
          <Panel title="Runtime Configuration">
            {config && (
              <div className="settings-grid">
                <Metric label="Model" value={config.model_path} />
                <label className="field">
                  <span>Confidence</span>
                  <input className="input" type="number" min="0.01" max="1" step="0.01" value={config.detector_confidence} onChange={(event) => setConfigNumber("detector_confidence", Number(event.target.value))} />
                </label>
                <label className="field">
                  <span>IoU</span>
                  <input className="input" type="number" min="0.01" max="1" step="0.01" value={config.detector_iou} onChange={(event) => setConfigNumber("detector_iou", Number(event.target.value))} />
                </label>
                <label className="field">
                  <span>Image Size</span>
                  <input className="input" type="number" min="160" max="1920" step="32" value={config.detector_imgsz} onChange={(event) => setConfigNumber("detector_imgsz", Number(event.target.value))} />
                </label>
                <label className="field">
                  <span>Frame Interval</span>
                  <input className="input" type="number" min="1" max="120" step="1" value={config.frame_interval} onChange={(event) => setConfigNumber("frame_interval", Number(event.target.value))} />
                </label>
                <button className="primary" onClick={saveSettings}>
                  <Settings size={18} /> Save Settings
                </button>
              </div>
            )}
          </Panel>
          <TrackingSettings />
          </>
        )}
      </section>
    </main>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function ChartPanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel chart">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function Empty({ text }: { text: string }) {
  return <div className="empty">{text}</div>;
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EventList({ events }: { events: EventRow[] }) {
  if (!events.length) return <Empty text="Events will appear after a processed video crosses a line or zone." />;
  return (
    <div className="list">
      {events.map((event) => (
        <div className="row" key={event.id}>
          <span>{event.event_type}</span>
          <strong>ID {event.track_id}</strong>
          <em>{event.timestamp.toFixed(1)}s</em>
        </div>
      ))}
    </div>
  );
}

function JobList({ jobs }: { jobs: Job[] }) {
  if (!jobs.length) return <Empty text="No processing jobs yet." />;
  return (
    <div className="list">
      {jobs.map((job) => (
        <div className="job" key={job.id}>
          <div className="job-head">
            <Video size={18} />
            <strong title={job.source_path}>{job.source_path.split(/[\\/]/).pop()}</strong>
            <span className={`status-badge ${job.status}`}>{job.status}</span>
          </div>
          {job.status !== "completed" && <progress value={job.progress} max={1} />}
          <div className="job-meta">
            <span><b>{job.entries}</b> entries</span>
            <span><b>{job.exits}</b> exits</span>
            {job.output_path && (
              <a href={`${API}/api/jobs/${job.id}/download`}>
                <Download size={15} /> Annotated MP4
              </a>
            )}
          </div>
          {job.error && <p className="error">{job.error}</p>}
        </div>
      ))}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<ReviewProvider><App /></ReviewProvider>);

