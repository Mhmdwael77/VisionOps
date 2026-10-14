# Appearance-Assisted Tracking

## Baseline And Opt-In

ByteTrack with a 30-update buffer remains the default. The original Retail Test
`model.track(..., tracker="bytetrack.yaml", persist=True)` path is retained for
that default, including its original one-second zone-loss grace. Existing persisted
analyses and detector weights are not rewritten. Settings stores tracking choices
in the backend database, independently of detector/geometry settings. Both retail
CLI/API jobs and generic video jobs consume the saved tracker snapshot.

Settings offers ByteTrack, BoT-SORT motion + GMC, and BoT-SORT + deep appearance.
The latter is experimental. Choosing it is not a recommendation or an accuracy
claim. Nothing automatically promotes a tracker after a comparison.

## Integration

This adapter is tested against and pins Ultralytics **8.4.7**. It uses upstream
BYTETracker/BOTSORT Kalman prediction, GMC, low-score association, and one-to-one
linear assignment. See [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/)
and the [versioned implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.7/ultralytics/trackers/bot_sort.py).

`NativeFeatures` follows the version's Detect-head pre-hook and NMS-aligned object
feature extraction. It never silently loads a classifier or alternative detector.
Unsupported native feature extraction fails explicitly. Native vectors are deep
detector features, **not a person-ReID-trained checkpoint**. No suitable dedicated
person-ReID checkpoint was found in the workspace. Optional ONNX encoders are
explicit local files; no model download is performed. Their contract is RGB,
256x128, ImageNet mean/std normalization, one crop per forward call, NxD output.
Operators must check a model's provenance, normalization, training domain, and
license before enabling it. The ONNX path has no installed real model to benchmark.

The scoped `GalleryTrack` initializes the upstream gallery before providing a
feature, avoiding the installed 8.4.7 constructor ordering bug. ReID mode also
retains lost hypotheses until expiry rather than applying upstream IoU-only
active/lost duplicate pruning, which can otherwise suppress a new contradictory
identity. Exclusive detection/active-ID assignment is still checked every frame.
ByteTrack and non-ReID BoT-SORT retain upstream pruning.

## Recovery Policy

Each BoT-SORT track has a bounded recent gallery. Reliable observations must meet
confidence, size/aspect ratio, image-edge, blur, and overlap checks. Clothing
descriptors are central torso/lower-body HSV histograms; deep vectors are normalized.
Unreliable observations can support upstream low-score motion rescue but never
update the appearance gallery. All thresholds are operator-adjustable.

Assisted recovery requires at least two recent reliable observations, a reliable
new observation, agreement in both deep appearance and HSV, predicted-box overlap,
size/center gates, and Kalman Mahalanobis distance <=9.4877. Candidate row/column
margins reject ambiguity and protect detections plausibly belonging to active
tracks. Clothing similarity alone cannot reconnect an ID. Ambiguity creates a new
hypothesis rather than forcing old-ID reuse. Active and low-score upstream stages
are not a guarantee against identity switches.

Buffers **10, 30, 60** mean processed update frames, not wall time. All comparison
frames are processed (stride 1), at the source's recorded FPS. Buffer seconds are
saved per variant as buffer/FPS. Generic jobs with frame skipping use FPS/stride.
Ultralytics normally normalizes `frame_rate/30 * track_buffer`; this adapter sets
the actual maximum update age explicitly so the requested buffer is not shortened
at this clip's approximately 13.093 FPS.

Native engine IDs flow directly into counters, zones, trajectories, and dwell.
There is no after-the-fact ID aliasing. Experimental zone grace covers the recovery
window. An observation gap censors an affected zone visit; dwell is still a tracked
time span, not verified uninterrupted presence. A line crossing across an
unobserved gap is provisional/censored and labeled accordingly. Initial occupancy
is unknown. No invisible detections or precise hidden movements are reconstructed.

Diagnostics distinguish current-frame active/lost hypotheses, cumulative same-ID
gap returns, and removed hypotheses. Removed counts include unconfirmed hypotheses
that never emitted an ID, so they can exceed the number of observed/output IDs.
An observation-gap return is not a verified correct-person recovery. An emitted
ID is not a unique visitor. Short tracks and within-ID gap runs are not measured
ground-truth fragmentation or ID switches.

## Reproduce

```powershell
python -m pytest -q
python scripts/run_tracking_experiment.py
```

Outputs live under `outputs/retail_store_test/tracking_experiment/<UTC-run-id>/`.
The runner freezes detector weights/hash, source hash, thresholds, geometry,
embedding source and library version. It detects the entire video once and stores
aligned detections/embeddings/timestamps in `detections.npz`. Each of the nine
mode/buffer combinations receives identical boxes/features. A tenth 30-frame ReID
ablation disables assisted lost-recovery gates while retaining quality-gated
native-feature association. This is an adapter ablation, not an unmodified upstream
ReID benchmark. Each variant saves an annotated H.264 MP4, three representative
frames, tracks, diagnostics, events CSV/JSON, zone visits, occupancy, and summary.
Videos are fully decoded to check frame count, dimensions, and FPS.

Previously generated detections can be reused only after validating hashes and
detector/encoder settings:

```powershell
python scripts/run_tracking_experiment.py --detection-cache outputs/retail_store_test/tracking_experiment/<run-id>
```

Replay FPS is measured decode + tracker + gallery/GMC + drawing + MP4 writing,
excluding detector and final encoding. End-to-end replay FPS also includes H.264
encoding/full decode verification. Charged-detector FPS is an explicitly labeled
estimate adding shared measured prediction time, not a separately timed online run.
Shared inference timing includes native-feature capture, charged equally to all
variants. Initial warmup and hardware contention affect timings.

The experiment is isolated from operational Overview/Retail results, not imported
as ten independent store videos. Settings displays its explicit experiment API,
diagnostics, comparison videos and report. Running an experiment does not change
the saved tracker or detector. Video launches are blocked during a controlled
comparison to avoid concurrent processing interference.

## Ground Truth

No complete annotated identities are available for this video, so true ID switches,
fragmentations, IDF1, MOTA, and identity/counting accuracy remain unavailable.
Fewer emitted IDs may mean better continuity **or false merges**; do not rank or
deploy trackers using this count alone. Geometry remains unreviewed.

For complete manually annotated identities, install the optional evaluator:

```powershell
pip install -r requirements-tracking-eval.txt
python scripts/run_tracking_experiment.py --ground-truth path/to/identities.csv
```

CSV header: `frame_index,person_id,x1,y1,x2,y2`. Frame indices are zero-based;
coordinates are original-video pixels. Annotate all evaluable visible people in
every frame; absent rows mean no ground-truth object, not an unlabeled frame.
Ignored regions and partial-annotation subsets are not supported by this simple
format. The evaluator uses [motmetrics](https://github.com/cheind/py-motmetrics)
with IoU >=0.5, and saves true switch/fragment counts, IDF1, MOTA, misses and false
positives. Synthetic evaluator tests are not evidence of video accuracy.

## Remaining Limits

Native detector features can be unstable under pose/light/scale changes. Conservative
gates may increase fragmentation or delay reporting rather than improve continuity.
Look-alike crowds, severe occlusion, clipping, camera motion and long disappearance
remain hard. GMC failures are counted and fall back to identity warp. Unit tests
include a known camera translation, not a moving-camera dataset evaluation.
Feature galleries cannot establish biometric identity or reconstruct hidden paths.
Dedicated person-ReID models and manually annotated identities are needed before
making a supported identity-quality comparison.
