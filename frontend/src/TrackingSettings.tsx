import { useEffect, useState } from "react";
import { Download, Play, RefreshCw, Save } from "lucide-react";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
type Tracking = {
  mode: string; lost_buffer_frames: number; gmc_method: string; recovery_enabled: boolean;
  embedding_backend: string; embedding_model: string | null; gallery_size: number; gallery_min_observations: number;
  gallery_max_age_frames: number; reliable_confidence: number; occlusion_iou: number; min_embedding_similarity: number;
  min_color_similarity: number; min_recovery_iou: number; max_center_distance: number; ambiguity_margin: number;
  track_high_thresh: number; track_low_thresh: number; new_track_thresh: number; match_thresh: number;
  reid_enabled: boolean; reid_model: string; reid_split_similarity: number; reid_merge_similarity: number;
  reid_min_link_similarity: number; reid_max_gap_seconds: number; reid_max_overlap_seconds: number; min_track_seconds: number;
};
type Diagnostics = { active: number; lost: number; recovered: number; removed: number; gmc_failures: number };
type ComparisonRow = { variant: string; generated_ids: number; buffer_seconds: number; processing_fps: number;
  detector_charged_fps_estimate: number; tracking_ms: { mean: number; p95: number }; entries: number; exits: number;
  last_diagnostics: Diagnostics; identity_metrics: Record<string, number> | null; video_fully_decoded: boolean };
type Experiment = { status: { state: string; phase?: string; variant?: string; processed_frames?: number; total_frames?: number;
  diagnostics?: Diagnostics; error?: string }; comparison: { relative_dir: string; results: ComparisonRow[];
  detector: { fps: number; frames: number; confidence: number; iou: number; imgsz: number; embedding_note: string; checkpoint: string; inference_ms: { mean: number } } } | null };

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(API+path, options);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

export function TrackingSettings() {
  const [config, setConfig] = useState<Tracking | null>(null);
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [selection, setSelection] = useState("bytetrack_30");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [loading, setLoading] = useState(true);
  async function refresh() {
    try { setExperiment(await request<Experiment>("/api/tracking-experiments")); setError(""); }
    catch (cause) { setError(String(cause)); }
    finally { setLoading(false); }
  }
  useEffect(() => {
    let active = true;
    request<{ tracking: Tracking }>("/api/config").then(data => { if (active) setConfig(data.tracking); }).catch(cause => { if (active) setError(String(cause)); });
    refresh();
    const timer = window.setInterval(refresh, 3000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);
  function change(key: keyof Tracking, value: string | number | boolean | null) {
    setConfig(current => current ? { ...current, [key]: value } : current);
    setDirty(true); setMessage("");
  }
  async function save() {
    setBusy(true);
    try {
      const result = await request<{ tracking: Tracking }>("/api/config", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tracking: config }) });
      setConfig(result.tracking); setDirty(false); setMessage("Tracker saved for future video jobs. Existing analyses are unchanged.");
    } catch (cause) { setError(String(cause)); } finally { setBusy(false); }
  }
  async function run() {
    setBusy(true);
    try { await request("/api/tracking-experiments/run", { method: "POST" }); setMessage("Controlled comparison queued."); await refresh(); }
    catch (cause) { setError(String(cause)); } finally { setBusy(false); }
  }
  const running = experiment?.status.state === "running";
  const comparison = experiment?.comparison;
  const chosen = comparison?.results.find(row => row.variant === selection);
  const diagnostics = running ? experiment?.status.diagnostics : chosen?.last_diagnostics;
  const url = (file: string) => `${API}/api/tracking-experiments/artifacts/${comparison?.relative_dir}/${file}`;
  const experimental = config?.mode !== "bytetrack";
  const reid = config?.mode === "botsort_reid";
  return <section className="tracking-settings">
    <div className="test-toolbar"><h2>Tracker</h2><button className="icon-button" title="Refresh tracker diagnostics" aria-label="Refresh tracker diagnostics" onClick={refresh}><RefreshCw size={18}/></button></div>
    {error && <p role="alert" className="error">Tracking API unavailable: {error}</p>}
    {message && <p role="status" className="notice">{message}</p>}
    {!config && <p className="muted">Loading tracking configuration...</p>}
    {config && <>
      <h3 className="settings-subtitle">Identity resolution</h3>
      <p className="muted">After tracking a recorded video, a person re-identification model (OSNet) cuts IDs that jumped between people, re-links IDs of the same person and hides IDs seen too briefly.</p>
      <div className="tracking-fields">
        <label className="tracking-check"><input type="checkbox" checked={config.reid_enabled} onChange={event => change("reid_enabled", event.target.checked)}/>Person re-identification</label>
        <label className="field"><span>Hide IDs shorter than (s)</span><input className="input" aria-label="Hide IDs shorter than" type="number" min={0} max={30} step={.5} value={config.min_track_seconds} onChange={event => change("min_track_seconds", Number(event.target.value))}/></label>
        <label className="field"><span>Max re-link gap (s)</span><input className="input" aria-label="Max re-link gap" type="number" disabled={!config.reid_enabled} min={0} max={600} step={5} value={config.reid_max_gap_seconds} onChange={event => change("reid_max_gap_seconds", Number(event.target.value))}/></label>
        {([ ["reid_merge_similarity", "Re-link similarity", .5, .99], ["reid_split_similarity", "Switch-cut similarity", 0, 1] ] as const).map(([key, label, min, max]) => <label className="field" key={key}><span>{label}: {config[key].toFixed(2)}</span><input aria-label={label} type="range" disabled={!config.reid_enabled} min={min} max={max} step=".01" value={config[key]} onChange={event => change(key, Number(event.target.value))}/></label>)}
      </div>
      <h3 className="settings-subtitle">Association</h3>
      <div className="tracking-fields">
        {([ ["new_track_thresh", "New track confidence", .05, .99], ["match_thresh", "Match tolerance (1 - IoU)", .3, .99], ["track_high_thresh", "High-score threshold", .05, .95] ] as const).map(([key, label, min, max]) => <label className="field" key={key}><span>{label}: {config[key].toFixed(2)}</span><input aria-label={label} type="range" min={min} max={max} step=".01" value={config[key]} onChange={event => change(key, Number(event.target.value))}/></label>)}
      </div>
      <h3 className="settings-subtitle">Engine</h3>
      <div className="tracking-fields">
        <label className="field"><span>Tracker</span><select aria-label="Tracker" value={config.mode} onChange={event => change("mode", event.target.value)}><option value="bytetrack">ByteTrack baseline</option><option value="botsort">BoT-SORT / motion + GMC</option><option value="botsort_reid">BoT-SORT + deep appearance</option></select></label>
        <label className="field"><span>Lost-track buffer</span><select aria-label="Lost-track buffer" value={config.lost_buffer_frames} onChange={event => change("lost_buffer_frames", Number(event.target.value))}>{[10, 30, 60, 90, 150].includes(config.lost_buffer_frames) ? null : <option value={config.lost_buffer_frames}>{config.lost_buffer_frames} frames</option>}{[10, 30, 60, 90, 150].map(value => <option key={value} value={value}>{value} update frames</option>)}</select></label>
        <label className="field"><span>Camera motion</span><select aria-label="Camera motion" disabled={!experimental} value={config.gmc_method} onChange={event => change("gmc_method", event.target.value)}><option value="none">None / fixed camera</option><option value="sparseOptFlow">Sparse optical flow</option><option value="orb">ORB</option><option value="ecc">ECC</option></select></label>
        <label className="tracking-check"><input type="checkbox" disabled={!reid} checked={config.recovery_enabled} onChange={event => change("recovery_enabled", event.target.checked)}/>Conservative appearance recovery</label>
        <label className="field"><span>Deep embedding source</span><select aria-label="Deep embedding source" disabled={!reid} value={config.embedding_backend} onChange={event => change("embedding_backend", event.target.value)}><option value="native">Native YOLO features</option><option value="onnx">Local person-ReID ONNX</option></select></label>
        {config.embedding_backend === "onnx" && <label className="field"><span>Local ONNX checkpoint</span><input className="input" aria-label="Local ONNX checkpoint" value={config.embedding_model ?? ""} onChange={event => change("embedding_model", event.target.value || null)}/></label>}
        {([ ["gallery_size", "Recent gallery size", 2, 100, 1], ["gallery_min_observations", "Minimum reliable observations", 2, 20, 1],
          ["gallery_max_age_frames", "Gallery age limit / frames", 10, 600, 1], ["reliable_confidence", "Reliable detection confidence", .25, .99, .01],
          ["occlusion_iou", "Occlusion overlap threshold", .05, .95, .01], ["max_center_distance", "Maximum center distance / box diagonal", .1, 2, .05],
          ["ambiguity_margin", "Association ambiguity margin", .01, .5, .01] ] as const).map(([key, label, min, max, step]) => <label className="field" key={key}><span>{label}</span><input className="input" aria-label={label} type="number" disabled={!experimental} min={min} max={max} step={step} value={config[key]} onChange={event => change(key, Number(event.target.value))}/></label>)}
        {([ ["min_embedding_similarity", "Minimum deep similarity", .5, .99], ["min_color_similarity", "Minimum clothing similarity", 0, 1], ["min_recovery_iou", "Minimum predicted-box overlap", .05, .9] ] as const).map(([key, label, min, max]) => <label className="field" key={key}><span>{label}: {config[key].toFixed(2)}</span><input aria-label={label} type="range" disabled={!reid} min={min} max={max} step=".01" value={config[key]} onChange={event => change(key, Number(event.target.value))}/></label>)}
      </div>
      <div className="test-actions"><button className="secondary" disabled={busy || !dirty} onClick={save}><Save size={16}/> Save Tracker</button><button className="secondary" disabled={busy || running || dirty} onClick={run}><Play size={16}/> Run Controlled Comparison</button></div>
    </>}
    <p className="test-warning">Experimental identity recovery is not validated customer identification. Native YOLO features are not a person-ReID-trained checkpoint. Fewer IDs do not prove better tracking. Detector weights are unchanged.</p>
    {loading && <p role="status" className="muted">Loading experiment diagnostics...</p>}
    {experiment && <p className="muted">Experiment: {experiment.status.state}{running ? ` / ${experiment.status.phase ?? ""} / ${experiment.status.variant ?? "shared detections"} / ${experiment.status.processed_frames ?? 0} of ${experiment.status.total_frames ?? "?"} frames` : ""}</p>}
    {experiment?.status.error && <p role="alert" className="error">{experiment.status.error}</p>}
    {diagnostics && <div className="overview-metrics">{(["active", "lost", "recovered", "removed"] as const).map(key => <div className="summary-metric" key={key}><span>{key[0].toUpperCase()+key.slice(1)}</span><strong data-tracker-diagnostic={key}>{diagnostics[key]}</strong><small>{running ? "Current replay frame" : "Selected video's final frame"}{key === "recovered" && " / cumulative gap returns, not verified identities"}</small></div>)}</div>}
    {comparison?.results.length ? <>
      <p className="muted">{comparison.detector.checkpoint} / confidence {comparison.detector.confidence} / IoU {comparison.detector.iou} / input {comparison.detector.imgsz}. {comparison.detector.frames} source frames / {comparison.detector.fps.toFixed(6)} updates per video second. {comparison.detector.embedding_note}. Detector inference mean {comparison.detector.inference_ms.mean.toFixed(2)} ms. Removed hypotheses include unconfirmed IDs that never emitted a detection.</p>
      <div className="overview-table-scroll"><table className="tracking-table"><thead><tr><th>Variant</th><th>Buffer / seconds</th><th>IDs, not people</th><th>Recovered gaps</th><th>Tracker mean / p95 ms</th><th>Replay FPS</th><th>Charged FPS estimate</th><th>Provisional in / out</th></tr></thead><tbody>{comparison.results.map(row => <tr key={row.variant}><td>{row.variant}</td><td>{row.buffer_seconds.toFixed(3)}</td><td>{row.generated_ids}</td><td>{row.last_diagnostics.recovered}</td><td>{row.tracking_ms.mean.toFixed(2)} / {row.tracking_ms.p95.toFixed(2)}</td><td>{row.processing_fps.toFixed(1)}</td><td>{row.detector_charged_fps_estimate.toFixed(1)}</td><td>{row.entries} / {row.exits}</td></tr>)}</tbody></table></div>
      <p className="muted">Replay FPS excludes detector and final encoding. Charged FPS adds the measured shared prediction time; it is an estimate, not a separately timed online pipeline. Identity switches and true fragmentation: {chosen?.identity_metrics ? JSON.stringify(chosen.identity_metrics) : "Unavailable without annotated ground-truth identities"}. Entrance geometry remains unreviewed.</p>
      <div className="test-toolbar"><label className="field"><span>Comparison video</span><select aria-label="Comparison video" value={selection} onChange={event => setSelection(event.target.value)}>{comparison.results.map(row => <option key={row.variant}>{row.variant}</option>)}</select></label>{experiment?.status.state === "completed" && <a className="secondary link" href={url("evaluation_report.md")}><Download size={16}/> Evaluation Report</a>}</div>
      {chosen && <video className="processed-video" crossOrigin="anonymous" key={url(selection)} controls preload="metadata" poster={url(`${selection}/frame_000000.jpg`)} src={url(`${selection}/annotated.mp4`)}/>}
    </> : !loading && !running && <p className="muted">No completed tracker comparisons are available.</p>}
  </section>;
}
