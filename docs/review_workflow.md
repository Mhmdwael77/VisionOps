# Real Video Review Workspace

The application runs locally at http://127.0.0.1:5173 with the API on port 8000.
Existing ByteTrack inference, weights, raw Overview, Retail Baseline and tracker
Settings remain available. No synthetic video, demo cameras or generated analytics
are used. The default Overview reads the saved review workspace, not hardcoded data.

## Data and Provenance

`POST /api/review/initialize` imports the latest completed retail analysis only when
the raw track export matches its run and the source video SHA-256 matches. It copies
raw observations into a separate SQLite `review_datasets` record. `raw_json` never
changes. Every successful edit stores a new `review_revisions` snapshot, note and
SHA-256 document fingerprint. Concurrent stale saves return HTTP 409.

The native editable JSON export contains `source_sha256`, `revision`, `metadata`,
`document`, `analytics` and `fingerprint`. The document contains:

- `annotations`: ID, zero-based frame index, person candidate ID, pixel `[x1,y1,x2,y2]`
  box, visibility, provenance, confidence and uncertainty.
- `identities`: labels, explicit identity confirmation and reviewer notes.
- `frame_reviews`: unreviewed, reviewed or ambiguous, with notes.
- `geometry`: normalized entrance endpoints, entry direction, dead band and zone
  polygon vertices, colors and separate review flags.
- `event_corrections`: accepted/rejected/reset decisions tied to evidence hashes.
- `manual_events`: reviewer-entered line/zone transitions with exact frame timestamps.

JSON and annotation/event/visit CSV exports are available in the application.
Importing JSON requires the same source hash and resets review certificates: an
import is a draft, not proof that a human reviewed the footage. Raw JSON exports
identify revision zero and have their own matching fingerprint and analytics.

## Editing

Video Analysis supports stepping, exact source-frame images while paused, timeline
seeking, playback speed, fullscreen, overlay toggles and immutable raw comparison.
Visible boxes can be moved/resized or edited numerically. Missing boxes can be
drawn and assigned a new unconfirmed ID. Occlusion records require a null box;
they never create invented visible detections. Deletes are audited.

Tracking Inspector shows appearance crops, observation spans and missing-frame
gaps. Merge requires nonoverlapping observations: one identity cannot have two
boxes in a frame. Split reassigns an explicit frame range to a new identity.
Identity changes invalidate affected frame review flags and generated event
decisions. Every edit requires a note. Confirmation is a reviewer assertion, not
automatic ground truth; ambiguous matches should remain unconfirmed.

Zone Analytics includes draggable geometry vertices, precise coordinate inputs,
direction/dead-band settings, colors and review flags. Geometry editing does not
change the raw inference configuration or silently rerun the detector.

Event decisions are accepted, rejected or reset. A wrong inferred event can be
rejected and replaced with a manual event at the corrected source frame. Manual
events cannot silently double-count an inferred crossing for the same identity
and direction. The baseline's conservative once-per-direction policy is retained;
legitimate repeated same-direction crossings need a policy extension and review.

## Metric Definitions

All review charts, exports and render overlays use `review_analytics.derive` over
one versioned annotation/event document, using source video timestamps.

- Provisional counts include nonrejected, nondeduplicated line events.
- Reviewed crossings require accepted events, reviewed geometry, a confirmed
  identity and reviewed observations on both sides of a transition. Gaps/censored
  boundaries do not become confirmed automatically.
- Visible occupancy counts only visible boxes whose bottom-center anchors are
  inside the geometry. It is not live, total-store, or inferred hidden occupancy.
- Zone visits pair nonrejected zone transitions. Orphan/repeated transitions are
  flagged incomplete. Initial observation, tracking loss, recovered gaps and video
  end censor a visit. Censored spans are labeled observed spans, not full dwell.
- Reviewed dwell pools complete visit durations, requiring confirmed boundaries
  and every covered frame reviewed. Missing reliable dwell is null, not zero.
- Tracker/person candidate IDs do not establish unique customers. Unknown initial
  occupancy, current live occupancy and unique visitors remain unavailable.

## Rendering and Verification

Rendering pins an immutable revision and checks the original source hash. Every
frame writes an overlay ledger with exact boxes, IDs, provenance, review state and
metrics. Trajectories only join consecutive visible observations. Boxes are not
interpolated or aesthetically smoothed away from the saved records. Raw jitter
therefore remains until corrected. Output is browser-compatible H.264 MP4.

All presentation outputs, source snapshots, ledgers, representative frames,
screenshots, validation reports and the pre-review SQLite backup are under
`outputs/retail_store_test/reviewed/`. The operational SQLite database remains in
its existing location, `visionops.db`.

```powershell
python -m pytest -q
python scripts/verify_review_ui.py
python scripts/verify_review_consistency.py
```

`verify_review_ui.py --verify-write` additionally re-saves the explicitly inspected
first-frame box unchanged and restores its ambiguous review state. It creates new
audited revisions, so render the new current revision before running consistency
checks. Ordinary UI verification is read-only.

`verify_review_restart.py capture` and `verify` compare exact API documents, raw
records, exports, history and video availability across actual service restarts;
`verify` takes explicit before/after backend and frontend process IDs.

Streaming routes release database sessions before sending video bytes. This is
important when seeking opens concurrent range requests; 20 open streams are tested
without exhausting the SQLite connection pool.

## Current Human Review Coverage

One missing, partially clipped right-edge person in frame zero was visually
inspected and added. Its identity is not linked to any later track. Frame zero
remains ambiguous because overlapping children/staff and identity continuity are
unresolved. Zero frames are fully reviewed; entrance/zone geometry is unreviewed.
This is a working correction and presentation system, not a completed ground-truth
annotation of the clip. No accuracy or identity-switch rate is claimed.

For production, add authentication, a dedicated job queue and video storage;
this local review application should not be exposed publicly as-is.
