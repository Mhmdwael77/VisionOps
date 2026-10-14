import { useEffect, useState } from "react";
import { Activity, Download, RefreshCw, Video } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
type Analysis = { id: string; source_name: string; source_path: string; status: string; started_at: string | null;
  completed_at: string | null; entries: number | null; exits: number | null; provisional: boolean;
  entrance_reviewed: boolean; zones_reviewed: boolean; output_available: boolean; error: string | null };
type Dwell = { visits: number; censored_visits: number; incomplete_visits: number; unknown_censoring_visits: number;
  complete_visits: number; measured_visits: number; average_dwell_seconds: number | null; average_uncensored_dwell_seconds: number | null };
type Zone = Dwell & { zone_name: string; tracker_id_count: number; historical_final_occupancy: number | null; historical_peak_occupancy: number | null };
type Event = { id: number; job_id: string; track_id: number; event_type: string; timestamp: number; label: string | null;
  censored: boolean | null; provisional: boolean; source_name: string };
type OverviewData = Dwell & { analysis_id: string; selected_analysis_ids: string[]; analyses: Analysis[]; history: Analysis[];
  entries: number | null; exits: number | null; event_count: number; completed_jobs: number; running_jobs: number;
  total_visitors: null; visitors_unavailable_reason: string; current_occupancy: null; occupancy_unavailable_reason: string;
  historical_visible_people: number | null; historical_peak_visible_people: number | null; tracker_id_count: number | null;
  line_counts_provisional: boolean; entrance_geometry_status: string; dwell_definition: string; recent_events: Event[];
  traffic: Array<{ bucket: string; entries: number; exits: number }>; zones: Zone[] };
const span = (value: number | null) => value == null ? "Unavailable" : `${value.toFixed(2)}s`;
const date = (value: string | null) => value ? new Date(/[Z+]/.test(value.slice(10)) ? value : `${value}Z`).toLocaleString() : "Time unavailable";

export function OverviewDashboard() {
  const [selection, setSelection] = useState(() => localStorage.getItem("visionops.analysis") ?? "latest");
  const [data, setData] = useState<OverviewData | null>(null);
  const [options, setOptions] = useState<Analysis[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const [updated, setUpdated] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    let fetching = false;
    const controller = new AbortController();
    setLoading(true);
    setData(null);
    setError("");
    async function refresh() {
      if (fetching) return;
      fetching = true;
      try {
        const response = await fetch(`${API}/api/analytics/overview?analysis_id=${encodeURIComponent(selection)}`, { signal: controller.signal });
        if (!response.ok) throw new Error(`Overview API returned ${response.status}: ${await response.text()}`);
        const result: OverviewData = await response.json();
        if (active) { setData(result); setOptions(result.analyses); setError(""); setUpdated(new Date().toLocaleTimeString()); }
      } catch (cause) {
        if (active && !controller.signal.aborted) setError(cause instanceof Error ? cause.message : String(cause));
      } finally { fetching = false; if (active) setLoading(false); }
    }
    refresh();
    const timer = window.setInterval(refresh, 3000);
    window.addEventListener("visionops:analysis-completed", refresh);
    return () => { active = false; controller.abort(); window.clearInterval(timer); window.removeEventListener("visionops:analysis-completed", refresh); };
  }, [selection, revision]);
  function choose(value: string) { localStorage.setItem("visionops.analysis", value); setSelection(value); }
  const selected = data?.analyses.find(analysis => analysis.id === data.selected_analysis_ids[0]);
  return <div className="overview-dashboard" data-analysis-ids={data?.selected_analysis_ids.join(",") ?? ""} data-selection={data?.analysis_id ?? ""}>
    <div className="test-toolbar">
      <div><h2>Processed Video Analytics</h2><p className="muted">{selection === "all" ? "Latest completed analysis per video" : selected?.source_name ?? "Latest completed analysis"}</p></div>
      <div className="test-actions"><label className="analysis-selector">Analysis<select aria-label="Analysis selection" value={selection} onChange={event => choose(event.target.value)}>
        <option value="latest">Latest completed analysis</option><option value="all">All videos (latest run each)</option>
        {options.map(analysis => <option key={analysis.id} value={analysis.id}>{analysis.source_name} / {date(analysis.started_at)}</option>)}
      </select></label><button className="icon-button" title="Refresh analytics" aria-label="Refresh analytics" onClick={() => setRevision(value => value+1)} disabled={loading}><RefreshCw size={18}/></button></div>
    </div>
    {loading && <div className="overview-state" role="status"><Activity size={20}/> Loading persisted analytics...</div>}
    {error && <div className="overview-api-error" role="alert"><strong>Analytics unavailable</strong><p>{error}</p>{data && <p>Showing the last successful response. Data may be stale.</p>}<button className="secondary" onClick={() => setRevision(value => value+1)}><RefreshCw size={16}/> Retry</button>{selection !== "latest" && <button className="secondary" onClick={() => choose("latest")}>Latest analysis</button>}</div>}
    {!loading && !error && data && !data.selected_analysis_ids.length && <section className="overview-state"><Video size={26}/><h3>No completed analyses</h3><p>Completed processing results will appear here.</p></section>}
    {data && data.selected_analysis_ids.length > 0 && <>
      <p className="test-warning">{data.line_counts_provisional ? "Provisional entry/exit counts" : "Entry/exit counts with reviewed geometry"}. Entrance geometry: {data.entrance_geometry_status}. {data.analyses.filter(a => data.selected_analysis_ids.includes(a.id)).some(a => !a.zones_reviewed) && "Zone geometry is unreviewed."}</p>
      <div className="overview-metrics">
        <SummaryMetric label="Entries" value={data.entries ?? "Unavailable"} sub={data.line_counts_provisional ? "Provisional line crossings" : "Observed line crossings"}/>
        <SummaryMetric label="Exits" value={data.exits ?? "Unavailable"} sub={data.line_counts_provisional ? "Provisional line crossings" : "Observed line crossings"}/>
        <SummaryMetric label="Mean tracked span" value={span(data.average_dwell_seconds)} sub={`${data.measured_visits} measured visits; ${data.censored_visits} censored`}/>
        <SummaryMetric label="Completed videos" value={data.completed_jobs} sub={`${data.event_count} video events / ${data.running_jobs} processing`}/>
      </div>
      <div className="overview-context">
        <SummaryMetric label="Tracker IDs" value={data.tracker_id_count ?? "Unavailable"} sub="IDs are scoped to each run, not unique customers"/>
        <SummaryMetric label="Unique customers" value="Unavailable" sub={data.visitors_unavailable_reason}/>
        <SummaryMetric label="Live occupancy" value="Unavailable" sub={data.occupancy_unavailable_reason}/>
        <SummaryMetric label="Visible at video end" value={data.historical_visible_people ?? "Unavailable"} sub={data.selected_analysis_ids.length === 1 ? `Historical tracked people / peak ${data.historical_peak_visible_people ?? "unavailable"}` : "Independent videos have no combined occupancy"}/>
      </div>
      <section className="overview-section"><div className="test-toolbar"><h3>Traffic</h3><p className="muted">Elapsed video time / 15-second intervals{selection === "all" ? " / combined relative timelines" : ""}</p></div>
        {data.traffic.length ? <ResponsiveContainer width="100%" height={250}><BarChart data={data.traffic}><CartesianGrid stroke="#303d42"/><XAxis dataKey="bucket" stroke="#a2adb5"/><YAxis allowDecimals={false} stroke="#a2adb5"/><Tooltip contentStyle={{ background: "#172023", border: "1px solid #465153", color: "#fff" }}/><Legend/><Bar dataKey="entries" name={data.line_counts_provisional ? "Provisional entries" : "Entries"} fill="#4ad69b"/><Bar dataKey="exits" name={data.line_counts_provisional ? "Provisional exits" : "Exits"} fill="#ffc665"/></BarChart></ResponsiveContainer> : <p className="muted">No timed traffic observations are available.</p>}
      </section>
      <section className="overview-section"><h3>Zone Analytics</h3><p className="muted dwell-definition">{data.dwell_definition}</p>
        {data.zones.length ? <div className="overview-table-scroll"><table><thead><tr><th>Zone</th><th>Track IDs</th><th>Visits</th><th>Video end / peak</th><th>Mean tracked span</th><th>Complete-only mean</th><th>Censored</th><th>Incomplete</th><th>Unknown censoring</th></tr></thead><tbody>{data.zones.map(zone => <tr key={zone.zone_name}><td>{zone.zone_name}</td><td>{zone.tracker_id_count}</td><td>{zone.visits}</td><td>{zone.historical_final_occupancy ?? "Unavailable"} / {zone.historical_peak_occupancy ?? "Unavailable"}</td><td>{span(zone.average_dwell_seconds)}</td><td>{span(zone.average_uncensored_dwell_seconds)}</td><td>{zone.censored_visits}</td><td>{zone.incomplete_visits}</td><td>{zone.unknown_censoring_visits}</td></tr>)}</tbody></table></div> : <p className="muted">No persisted zone visits or configured zones for this analysis.</p>}
        <p className="muted">Zone end/peak counts are historical visible tracks, not live occupancy; unavailable across independent videos.</p>
        <p className="muted">Overall complete-only mean: {span(data.average_uncensored_dwell_seconds)} across {data.complete_visits} visits. {data.incomplete_visits} incomplete visits excluded.</p>
      </section>
      <section className="overview-section"><div className="test-toolbar"><h3>Recent Events</h3><span className="muted">Video timestamps, newest analysis first</span></div>
        {data.recent_events.length ? <div className="overview-table-scroll"><table><thead><tr><th>Video</th><th>Video time</th><th>Track ID</th><th>Event</th><th>Region</th><th>Status</th></tr></thead><tbody>{data.recent_events.map(event => <tr key={event.id}><td>{event.source_name}</td><td>{event.timestamp.toFixed(2)}s</td><td>{event.track_id}</td><td>{event.event_type.replace(/_/g, " ")}</td><td>{event.label ?? "-"}</td><td>{event.provisional && "Provisional "}{event.censored ? "Censored" : ""}</td></tr>)}</tbody></table></div> : <p className="muted">No recorded events for the selected completed analysis.</p>}
      </section>
    </>}
    {data && <section className="overview-section"><h3>Processing History</h3>{data.history.length ? <div className="overview-table-scroll"><table><thead><tr><th>Video</th><th>Started</th><th>Status</th><th>Entries / Exits</th><th>Analysis</th></tr></thead><tbody>{data.history.map(analysis => <tr key={analysis.id}><td>{analysis.source_name}</td><td>{date(analysis.started_at)}</td><td>{analysis.status}{analysis.error && <span className="error">: {analysis.error}</span>}</td><td>{analysis.entries == null ? "Unavailable" : `${analysis.entries} / ${analysis.exits}${analysis.provisional ? " provisional" : ""}`}</td><td className="history-actions">{analysis.status === "completed" && <button className="time-link" onClick={() => choose(analysis.id)}>View analysis</button>}{analysis.output_available && <a href={`${API}/api/jobs/${analysis.id}/download`} title="Download processed video"><Download size={16}/></a>}</td></tr>)}</tbody></table></div> : <p className="muted">No processing jobs have been recorded.</p>}</section>}
    {updated && <p className="muted">Last successful refresh {updated}</p>}
  </div>;
}

function SummaryMetric({ label, value, sub }: { label: string; value: string | number; sub: string }) {
  return <div className="summary-metric" data-metric={label}><span>{label}</span><strong>{value}</strong><small>{sub}</small></div>;
}
