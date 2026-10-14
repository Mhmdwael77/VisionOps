import { useEffect, useRef, useState } from "react";
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChevronLeft, ChevronRight, Download, Play, Save, Video, Waypoints } from "lucide-react";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const artifact = (name: string) => `${API}/api/retail-test/artifacts/${name}`;
type Pair = [number, number];
type Geometry = {
  source: string; output_dir: string; model_path: string; fallback_model: string;
  confidence: number; iou: number; imgsz: number; device: string | null;
  trajectory_points: number; missing_grace_seconds: number; zones_reviewed: boolean;
  tracking?: { mode: string; lost_buffer_frames: number };
  zones: Array<{ name: string; polygon: Pair[] }>;
  counting_line: { name: string; start: Pair; end: Pair; entry_direction: string; min_displacement: number; reviewed: boolean };
};
type Timeline = { timestamp: number; visible_people: number; entries: number; exits: number; zones: Record<string, number> };
type Evidence = { frame_index: number; timestamp: number; source: string; annotated: string; tags: string[]; tracks: number[] };
type TestEvent = { id: number; timestamp: number; track_id: number; event_type: string; zone_name?: string; line_name?: string; dwell_seconds?: number; censored: boolean; reason: string };
type TestSummary = {
  tracker?: { label?: string; diagnostics?: { active: number; lost: number; recovered: number; removed: number } };
  run_id: string; configuration: Geometry;
  model: { path: string; kind: string; fallback_used: boolean };
  source: { decoded_frames: number; duration_seconds: number; width: number; height: number };
  environment: { gpu: string | null; device: string };
  performance: { processing_fps: number; end_to_end_fps: number; inference_ms: { mean: number; p95: number } };
  analytics: { unique_track_ids: number; entries: number; exits: number; peak_visible_people: number; event_count: number;
    zones: Array<{ name: string; visits: number; censored_visits: number; incomplete_visits?: number; unique_track_ids: number; average_dwell_seconds: number | null; average_uncensored_dwell_seconds: number | null; peak_occupancy: number }> };
};
type Results = { status: { state: string; progress: number; processed_frames?: number; error?: string; tracker_diagnostics?: { active: number; lost: number; recovered: number; removed: number } }; summary: TestSummary | null;
  timeline: Timeline[]; events: TestEvent[]; evidence: Evidence[];
  manual_review: { observations: Array<{ category: string; timestamp: number; finding: string; status: string; evidence: string }> } | null };

async function json<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}/api/retail-test${path}`, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export function RetailTest() {
  const [results, setResults] = useState<Results | null>(null);
  const [config, setConfig] = useState<Geometry | null>(null);
  const [view, setView] = useState("results");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [time, setTime] = useState(0);
  const [evidenceIndex, setEvidenceIndex] = useState(0);
  const [evidenceMode, setEvidenceMode] = useState<"source" | "annotated">("annotated");
  const [eventFilter, setEventFilter] = useState("all");
  const video = useRef<HTMLVideoElement>(null);
  const summary = results?.summary;
  const running = ["running", "encoding", "queued"].includes(results?.status.state ?? "");

  useEffect(() => {
    let mounted = true;
    let previousState = "";
    const refresh = () => json<Results>("").then(data => { if (mounted) {
      setResults(data);
      if (data.status.state === "completed" && previousState !== "completed") window.dispatchEvent(new Event("visionops:analysis-completed"));
      previousState = data.status.state;
    } }).catch(error => { if (mounted) setMessage(error.message); });
    refresh();
    json<Geometry>("/config").then(data => { if (mounted) setConfig(data); }).catch(error => { if (mounted) setMessage(error.message); });
    const timer = window.setInterval(refresh, 3000);
    return () => { mounted = false; window.clearInterval(timer); };
  }, []);

  async function save() {
    if (!config) return;
    setBusy(true);
    try {
      setConfig(await json<Geometry>("/config", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(config) }));
      setDirty(false);
      setMessage("Geometry saved. Rerun to apply it to the video.");
    } catch (error) { setMessage(String(error)); } finally { setBusy(false); }
  }
  async function run() {
    setBusy(true);
    try {
      await json("/run", { method: "POST" });
      setResults(current => current ? { ...current, summary: null, status: { state: "queued", progress: 0 } } : current);
      setMessage("");
    } catch (error) { setMessage(String(error)); } finally { setBusy(false); }
  }
  function seek(timestamp: number) {
    if (video.current) video.current.currentTime = timestamp;
    setTime(timestamp);
  }
  const priorReadings = results?.timeline.filter(row => row.timestamp <= time) ?? [];
  const current = priorReadings[priorReadings.length - 1];
  const selectedEvidence = results?.evidence[evidenceIndex];
  const filteredEvents = results?.events.filter(row => eventFilter === "all" || (eventFilter === "line" ? Boolean(row.line_name) : row.zone_name === eventFilter)) ?? [];
  const changedConfig = summary && config && Object.entries(config).some(([key, value]) => {
    if (key === "tracking" && !("tracking" in summary.configuration)) {
      const tracker = value as { mode: string; lost_buffer_frames: number };
      return tracker.mode !== "bytetrack" || tracker.lost_buffer_frames !== 30;
    }
    return JSON.stringify(value) !== JSON.stringify(summary.configuration[key as keyof Geometry]);
  });
  const trackerDiagnostics = running ? results?.status.tracker_diagnostics : summary?.tracker?.diagnostics;

  return <div className="retail-test">
    <div className="test-toolbar">
      <div><h2>Retail Store Test</h2><p className="muted">retail_store.mp4 <span className="test-state">{results?.status.state ?? "connecting"}</span></p></div>
      <div className="test-actions">
        <div className="segments" aria-label="Test views">
          <button className={view === "results" ? "selected" : ""} onClick={() => setView("results")}><Video size={16}/> Results</button>
          <button className={view === "geometry" ? "selected" : ""} onClick={() => setView("geometry")}><Waypoints size={16}/> Geometry</button>
        </div>
        <button className="secondary" title="Run the complete retail video" onClick={run} disabled={busy || running || dirty || !config}><Play size={16}/> Run Test</button>
      </div>
    </div>
    {message && <p className="notice" role="status">{message}</p>}
    {results?.status.error && <p className="error" role="alert">{results.status.error}</p>}
    {running && <div><progress max={1} value={results?.status.progress ?? 0}/><p className="muted">{results?.status.processed_frames ?? 0} frames processed</p></div>}
    {trackerDiagnostics && <div className="overview-metrics">{(["active", "lost", "recovered", "removed"] as const).map(key => <TestStat key={key} label={key[0].toUpperCase()+key.slice(1)} value={trackerDiagnostics[key]} detail={key === "recovered" ? "Cumulative observation-gap returns, not verified identities" : running ? "Current video-processing frame" : "Historical final frame"}/>)}</div>}
    {view === "geometry" && config && <GeometryEditor config={config} disabled={running || busy} onChange={value => { setConfig(value); setDirty(true); }} onSave={save} dirty={dirty}/>}
    {view === "results" && !summary && !running && <p className="muted">No completed retail test is available.</p>}
    {view === "results" && summary && <>
      <div className="test-warning" role="status">{summary.configuration.counting_line.reviewed ? "Entrance geometry reviewed" : "Entrance line unreviewed: entry and exit counts are provisional"}. {summary.configuration.zones_reviewed ? "Zone geometry reviewed." : "Cashier and shelf polygons are provisional."}</div>
      {changedConfig && <p className="test-warning">Saved configuration differs from this run.</p>}
      <div className="test-stats">
        <TestStat label="Processing" value={`${summary.performance.processing_fps.toFixed(1)} FPS`} detail={`${summary.performance.end_to_end_fps.toFixed(1)} FPS end to end`}/>
        <TestStat label="YOLO latency" value={`${summary.performance.inference_ms.mean.toFixed(1)} ms`} detail={`p95 ${summary.performance.inference_ms.p95.toFixed(1)} ms`}/>
        <TestStat label="Tracker IDs" value={summary.analytics.unique_track_ids} detail="Not unique people"/>
        <TestStat label="Line crossings" value={`${summary.analytics.entries} in / ${summary.analytics.exits} out`} detail={summary.configuration.counting_line.reviewed ? "Reviewed geometry" : "Provisional entrance geometry"}/>
      </div>
      <div className="test-playback">
        <div>
          <video ref={video} key={summary.run_id} className="processed-video" crossOrigin="anonymous" controls preload="metadata" poster={artifact(results?.evidence[0]?.annotated ?? "source_frame_0000.jpg")} src={`${artifact("annotated.mp4")}?run=${encodeURIComponent(summary.run_id)}`} onTimeUpdate={event => setTime(event.currentTarget.currentTime)}/>
          <div className="playback-caption"><span>{summary.source.decoded_frames} frames / {summary.source.duration_seconds.toFixed(2)}s</span><span>{summary.model.kind} / {summary.tracker?.label ?? "ByteTrack"}</span></div>
        </div>
        <div className="test-observations">
          <h3>At {time.toFixed(2)}s</h3>
          <TestStat label="Visible tracked people" value={current?.visible_people ?? 0} detail={`Peak ${summary.analytics.peak_visible_people}`}/>
          {summary.analytics.zones.map(zone => <div className="zone-reading" key={zone.name}>
            <h3>{zone.name}</h3><strong>{current?.zones[zone.name] ?? 0} visible</strong>
            <p>{zone.average_dwell_seconds == null ? "Unavailable" : `${zone.average_dwell_seconds.toFixed(2)}s`} mean tracked span</p>
            <p className="muted">Complete-only: {zone.average_uncensored_dwell_seconds == null ? "Unavailable" : `${zone.average_uncensored_dwell_seconds.toFixed(2)}s`}</p>
            <p className="muted">{zone.visits} visits / {zone.censored_visits} censored / {zone.incomplete_visits ?? 0} incomplete</p>
          </div>)}
          <p className="muted">{summary.environment.gpu ?? summary.environment.device}</p>
        </div>
      </div>
      <div className="test-charts">
        <section><h3>Visible Zone Occupancy</h3><ResponsiveContainer width="100%" height={230}>
          <LineChart data={results?.timeline}><CartesianGrid stroke="#30383d"/><XAxis dataKey="timestamp" tickFormatter={v => `${Math.round(v)}s`} stroke="#a2adb5"/><YAxis allowDecimals={false} stroke="#a2adb5"/><Tooltip/>
            {summary.analytics.zones.map((zone, i) => <Line key={zone.name} name={zone.name} dataKey={row => row.zones[zone.name]} type="stepAfter" dot={false} stroke={i ? "#ffc665" : "#4ad69b"} isAnimationActive={false}/>)}</LineChart>
        </ResponsiveContainer></section>
        <section><h3>Visible People</h3><ResponsiveContainer width="100%" height={230}>
          <AreaChart data={results?.timeline}><CartesianGrid stroke="#30383d"/><XAxis dataKey="timestamp" tickFormatter={v => `${Math.round(v)}s`} stroke="#a2adb5"/><YAxis allowDecimals={false} stroke="#a2adb5"/><Tooltip/><Area dataKey="visible_people" name="Visible tracked people" stroke="#52bfe1" fill="#52bfe122" isAnimationActive={false}/></AreaChart>
        </ResponsiveContainer></section>
      </div>
      <section className="test-evidence"><div className="test-toolbar"><h3>Detection Evidence</h3><div className="segments">
        <button className={evidenceMode === "source" ? "selected" : ""} onClick={() => setEvidenceMode("source")}>Source</button><button className={evidenceMode === "annotated" ? "selected" : ""} onClick={() => setEvidenceMode("annotated")}>Annotated</button>
      </div></div>
        {selectedEvidence && <>
          <div className="evidence-controls"><button className="icon-button" title="Previous evidence frame" disabled={evidenceIndex === 0} onClick={() => setEvidenceIndex(i => i-1)}><ChevronLeft size={18}/></button>
            <select aria-label="Evidence frame" value={evidenceIndex} onChange={event => setEvidenceIndex(Number(event.target.value))}>{results?.evidence.map((frame, i) => <option value={i} key={frame.frame_index}>{frame.timestamp.toFixed(2)}s / frame {frame.frame_index} / {frame.tags.join(", ").replace(/_/g, " ")}</option>)}</select>
            <button className="icon-button" title="Next evidence frame" disabled={evidenceIndex >= (results?.evidence.length ?? 0)-1} onClick={() => setEvidenceIndex(i => i+1)}><ChevronRight size={18}/></button>
            <button className="icon-button" title="Seek video to evidence frame" onClick={() => seek(selectedEvidence.timestamp)}><Play size={18}/></button>
          </div>
          <img className="evidence-image" src={artifact(selectedEvidence[evidenceMode])} alt={`${evidenceMode} retail frame at ${selectedEvidence.timestamp.toFixed(2)} seconds`}/>
          <p className="muted">{selectedEvidence.tags.join(", ").replace(/_/g, " ")} / IDs: {selectedEvidence.tracks.join(", ")}</p>
        </>}
      </section>
      <section><div className="test-toolbar"><h3>Video Events ({filteredEvents.length})</h3><select aria-label="Event filter" value={eventFilter} onChange={event => setEventFilter(event.target.value)}><option value="all">All events</option><option value="line">Line crossings</option>{summary.analytics.zones.map(zone => <option key={zone.name}>{zone.name}</option>)}</select></div>
        <div className="event-table-scroll"><table><thead><tr><th>Video Time</th><th>ID</th><th>Event</th><th>Region</th><th>Dwell</th><th>Observation</th></tr></thead><tbody>{filteredEvents.map(event => <tr key={event.id} onClick={() => seek(event.timestamp)}>
          <td><button className="time-link" title="Seek to event">{event.timestamp.toFixed(2)}s</button></td><td>{event.track_id}</td><td>{event.event_type.replace(/_/g, " ")}</td><td>{event.zone_name ?? event.line_name}</td><td>{event.dwell_seconds == null ? "-" : `${event.dwell_seconds.toFixed(2)}s`}</td><td>{event.censored ? `Censored: ${event.reason.replace(/_/g, " ")}` : event.reason?.replace(/_/g, " ")}</td>
        </tr>)}</tbody></table></div>
      </section>
      {results?.manual_review && <section><h3>Sampled Visual Review</h3>{results.manual_review.observations.map((observation, i) => <p className="review-observation" key={i}><strong>{observation.category} / {observation.timestamp.toFixed(2)}s</strong> {observation.finding} <span className="muted">({observation.status})</span></p>)}</section>}
      <section className="test-exports"><h3>Run Artifacts</h3><div className="test-actions">{[["events.csv", "CSV"], ["events.json", "JSON"], ["frame_metrics.csv", "Frame metrics"], ["test_report.md", "Test report"], ["annotated.mp4", "MP4"]].map(([file, label]) => <a className="secondary" href={artifact(file)} download={file} key={file}><Download size={16}/>{label}</a>)}</div><p className="muted">No ground truth accuracy measured. Censored dwell spans end at the last observed video timestamp.</p></section>
    </>}
  </div>;
}

function TestStat({ label, value, detail }: { label: string; value: string | number; detail: string }) {
  return <div className="test-stat"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function GeometryEditor({ config, disabled, dirty, onChange, onSave }: { config: Geometry; disabled: boolean; dirty: boolean; onChange: (value: Geometry) => void; onSave: () => void }) {
  const [region, setRegion] = useState(-1);
  const [drag, setDrag] = useState<number | null>(null);
  const geometry = useRef<SVGSVGElement>(null);
  const points = region < 0 ? [config.counting_line.start, config.counting_line.end] : config.zones[region].polygon;
  const colors = ["#4ad69b", "#ffc665"];
  function move(event: React.PointerEvent<SVGSVGElement>) {
    if (drag == null || disabled || !geometry.current) return;
    const rect = geometry.current.getBoundingClientRect();
    const pair: Pair = [Math.max(0, Math.min(1, (event.clientX-rect.left)/rect.width)), Math.max(0, Math.min(1, (event.clientY-rect.top)/rect.height))];
    if (region < 0) onChange({ ...config, counting_line: { ...config.counting_line, [drag === 0 ? "start" : "end"]: pair, reviewed: false } });
    else onChange({ ...config, zones_reviewed: false, zones: config.zones.map((zone, index) => index === region ? { ...zone, polygon: zone.polygon.map((p, i) => i === drag ? pair : p) } : zone) });
  }
  return <section className="geometry-editor"><div className="test-toolbar"><select aria-label="Geometry region" disabled={disabled} value={region} onChange={event => setRegion(Number(event.target.value))}><option value={-1}>Entrance line</option>{config.zones.map((zone, i) => <option key={zone.name} value={i}>{zone.name}</option>)}</select><button className="primary" disabled={disabled || !dirty} onClick={onSave}><Save size={16}/> Save Geometry</button></div>
    <div className="geometry-image"><img src={artifact("source_frame_0000.jpg")} alt="Retail store geometry reference"/><svg ref={geometry} viewBox="0 0 1270 720" onPointerMove={move} onPointerUp={() => setDrag(null)} onPointerCancel={() => setDrag(null)}>
      {config.zones.map((zone, i) => <polygon key={zone.name} points={zone.polygon.map(([x, y]) => `${x*1270},${y*720}`).join(" ")} fill={`${colors[i%2]}22`} stroke={colors[i%2]} strokeWidth="3"/>)}
      <line x1={config.counting_line.start[0]*1270} y1={config.counting_line.start[1]*720} x2={config.counting_line.end[0]*1270} y2={config.counting_line.end[1]*720} stroke="#ff7282" strokeWidth="4"/>
      {points.map(([x, y], i) => <circle key={i} cx={x*1270} cy={y*720} r="11" fill="white" stroke="#222" strokeWidth="2" className="geometry-handle" onPointerDown={event => { if (!disabled) { geometry.current?.setPointerCapture(event.pointerId); setDrag(i); } }}><title>{region < 0 ? (i ? "Line end" : "Line start") : `Vertex ${i+1}`}</title></circle>)}
    </svg></div>
    <div className="geometry-options"><label>Entry direction<select disabled={disabled} value={config.counting_line.entry_direction} onChange={event => onChange({ ...config, counting_line: { ...config.counting_line, entry_direction: event.target.value, reviewed: false } })}><option value="positive_to_negative">Positive to negative side</option><option value="negative_to_positive">Negative to positive side</option></select></label>
      <label>Line dead band (pixels)<input type="number" min="0" max="200" disabled={disabled} value={config.counting_line.min_displacement} onChange={event => onChange({ ...config, counting_line: { ...config.counting_line, min_displacement: Number(event.target.value) } })}/></label>
      <label className="check-label"><input type="checkbox" disabled={disabled} checked={config.counting_line.reviewed} onChange={event => onChange({ ...config, counting_line: { ...config.counting_line, reviewed: event.target.checked } })}/>Entrance location and direction reviewed</label>
      <label className="check-label"><input type="checkbox" disabled={disabled} checked={config.zones_reviewed} onChange={event => onChange({ ...config, zones_reviewed: event.target.checked })}/>Cashier and shelf polygons reviewed</label>
    </div>
    <p className="test-warning">{config.counting_line.reviewed ? "Entrance geometry marked reviewed." : "Entrance location and direction await human review."} {dirty ? "Unsaved changes." : "Saved configuration."}</p>
  </section>;
}
