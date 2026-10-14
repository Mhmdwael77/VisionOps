"""One deterministic annotation/event ledger for the editor, charts and renderer."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json

from visionops.vision.counting import CountingLineConfig, CrossingDirection, LineCrossingCounter
from visionops.vision.geometry import LineSegment, Point, point_in_polygon
from visionops.vision.zones import ZoneConfig, ZoneMonitor


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def frame_annotations(document):
    frames = defaultdict(list)
    for annotation in document["annotations"]:
        frames[annotation["frame_index"]].append(annotation)
    return frames


def derive(document, metadata):
    width, height = metadata["width"], metadata["height"]
    timestamps = metadata["timestamps"]
    frames = frame_annotations(document)
    geometry = document["geometry"]
    line = geometry["counting_line"]
    def point(pair):
        return Point(pair[0]*width, pair[1]*height)
    counter = LineCrossingCounter(CountingLineConfig(line["name"], LineSegment(point(line["start"]), point(line["end"])),
        CrossingDirection(line["entry_direction"]), line["min_displacement"]))
    zones = ZoneMonitor([ZoneConfig(zone["name"], [point(pair) for pair in zone["polygon"]]) for zone in geometry["zones"]])
    events, timeline = [], []
    last_seen, track_stats = {}, {}
    labels = {key: value["label"] for key, value in document["identities"].items()}
    review = document["frame_reviews"]
    identity_annotations = defaultdict(list)
    for annotation in document["annotations"]:
        identity_annotations[annotation["person_id"]].append(annotation)
    track_evidence = {identity: fingerprint(annotations) for identity, annotations in identity_annotations.items()}
    def append_event(value, frame, kind):
        identity = value.pop("track_id")
        region = value.get("zone_name", value.get("line_name"))
        value.update(person_id=identity, frame_index=frame, kind=kind, region=region)
        value["id"] = fingerprint([kind, identity, region, value["event_type"], frame])[:24]
        value["evidence_hash"] = fingerprint({"event": value, "geometry": geometry, "observations": track_evidence[identity]})
        value["review_status"], value["origin"] = "unreviewed", "annotation_derived"
        value["person_label"] = labels.get(identity, identity)
        events.append(value)
    for index, timestamp in enumerate(timestamps):
        visible = [a for a in frames[index] if a["visibility"] == "visible"]
        active = {a["person_id"] for a in visible}
        for a in visible:
            identity = a["person_id"]
            bbox = a["bbox"]
            anchor = Point((bbox[0]+bbox[2])/2, bbox[3])
            gap = identity in last_seen and index-last_seen[identity] > 1
            if gap:
                zones.mark_recovered_gap(identity)
            crossing = counter.update(identity, anchor, timestamp)
            if crossing:
                append_event({**asdict(crossing), "censored": gap, "reason": "unobserved_gap" if gap else "observed_crossing"}, index, "line")
            for event in zones.update(identity, anchor, timestamp):
                append_event(asdict(event), index, "zone")
            last_seen[identity] = index
            stat = track_stats.setdefault(identity, {"person_id": identity, "label": labels[identity], "first_frame": index,
                "last_frame": index, "observations": 0, "gaps": 0, "reviewed_observations": 0, "segments": [], "thumbnail_annotation": a["id"]})
            stat["last_frame"], stat["observations"] = index, stat["observations"]+1
            stat["gaps"] += gap
            stat["reviewed_observations"] += review[index]["state"] == "reviewed"
            if not stat["segments"] or index > stat["segments"][-1][1]+1:
                stat["segments"].append([index, index])
            else:
                stat["segments"][-1][1] = index
        for event in zones.expire_missing(active, timestamp, metadata.get("missing_grace_seconds", 1)):
            append_event(asdict(event), last_seen[event.track_id], "zone")
        timeline.append({"frame_index": index, "timestamp": timestamp, "visible": len(active), "zones": zones.occupancy(active),
                         "review_state": review[index]["state"]})
    for event in zones.close_open_visits(timestamps[-1]):
        append_event(asdict(event), last_seen[event.track_id], "zone")
    events.sort(key=lambda event: (event["timestamp"], event["event_type"] == "zone_exit", event["id"]))
    corrections = document["event_corrections"]
    applied, stale = set(), []
    for event in events:
        correction = corrections.get(event["id"])
        if correction:
            applied.add(event["id"])
            if correction.get("evidence_hash") != event["evidence_hash"]:
                stale.append(event["id"])
            else:
                event["review_status"] = correction["status"]
                event["note"] = correction.get("note", "")
    stale.extend(key for key in corrections if key not in applied)
    for event in document["manual_events"]:
        value = {**event, "origin": "manual_event", "person_label": labels.get(event["person_id"], event["person_id"])}
        value["timestamp"] = timestamps[event["frame_index"]]
        events.append(value)
    events.sort(key=lambda event: (event["timestamp"], event["event_type"] == "zone_exit", event["id"]))
    accepted = [event for event in events if event["review_status"] != "rejected"]
    line_events = [event for event in accepted if event["kind"] == "line"]
    # An operator-entered event cannot silently duplicate an inferred crossing.
    deduped, crossing_keys, duplicate_events = [], set(), []
    for event in sorted(line_events, key=lambda item: (item["origin"] != "manual_event", item["timestamp"])):
        key = event["person_id"], event["event_type"]
        if key in crossing_keys:
            duplicate_events.append(event["id"])
        else:
            crossing_keys.add(key)
            deduped.append(event)
    line_events = sorted(deduped, key=lambda item: item["timestamp"])
    for event in events:
        event["suppressed_duplicate"] = event["id"] in duplicate_events
        geometry_reviewed = line["reviewed"] if event["kind"] == "line" else next(zone.get("reviewed", False) for zone in geometry["zones"] if zone["name"] == event["region"])
        index = event["frame_index"]
        interval_reviewed = review[index]["state"] == "reviewed"
        if not event.get("censored"):
            previous = next((i for i in range(index-1, -1, -1)
                             if any(a["person_id"] == event["person_id"] and a["visibility"] == "visible" for a in frames[i])), None)
            interval_reviewed = previous is not None and all(review[i]["state"] == "reviewed" for i in range(previous, index+1))
        event["confirmed"] = bool(event["review_status"] == "accepted" and geometry_reviewed and not event.get("censored")
                                  and interval_reviewed and not event["suppressed_duplicate"]
                                  and document["identities"][event["person_id"]]["confirmed"])
    visits, incomplete = [], []
    pending = {}
    for event in [item for item in accepted if item["kind"] == "zone"]:
        key = event["person_id"], event["region"]
        if event["event_type"] == "zone_entry":
            if key in pending:
                incomplete.append(pending[key]["id"])
            pending[key] = event
        elif key not in pending:
            incomplete.append(event["id"])
        else:
            entry = pending.pop(key)
            if event["timestamp"] < entry["timestamp"]:
                incomplete.extend([entry["id"], event["id"]])
                continue
            covered = all(review[i]["state"] == "reviewed" for i in range(entry["frame_index"], event["frame_index"]+1))
            censored = entry.get("censored", False) or event.get("censored", False)
            visits.append({"person_id": key[0], "zone_name": key[1], "entered_at": entry["timestamp"], "exited_at": event["timestamp"],
                           "dwell_seconds": event["timestamp"]-entry["timestamp"], "censored": censored,
                           "confirmed": entry["confirmed"] and event["confirmed"] and covered and not censored,
                           "entry_event": entry["id"], "exit_event": event["id"]})
    incomplete.extend(event["id"] for event in pending.values())
    def mean(items):
        return sum(items)/len(items) if items else None
    zone_stats = []
    for zone in geometry["zones"]:
        rows = [v for v in visits if v["zone_name"] == zone["name"]]
        zone_stats.append({"name": zone["name"], "color": zone["color"], "geometry_reviewed": zone.get("reviewed", False),
            "visits": len(rows), "censored": sum(row["censored"] for row in rows), "confirmed_visits": sum(row["confirmed"] for row in rows),
            "mean_span": mean([row["dwell_seconds"] for row in rows]), "reviewed_dwell": mean([row["dwell_seconds"] for row in rows if row["confirmed"]]),
            "peak_visible": max(row["zones"][zone["name"]] for row in timeline), "final_visible": timeline[-1]["zones"][zone["name"]]})
    traffic = [{"start": start, "label": f"{start//60:02d}:{start%60:02d}", "entries": 0, "exits": 0, "reviewed_entries": 0, "reviewed_exits": 0}
               for start in range(0, int(timestamps[-1])+1, 15)]
    for event in line_events:
        bucket = traffic[int(event["timestamp"]//15)]
        bucket["entries" if event["event_type"] == "entry" else "exits"] += 1
        if event["confirmed"]:
            bucket["reviewed_entries" if event["event_type"] == "entry" else "reviewed_exits"] += 1
    cumulative = Counter()
    crossings_by_frame = defaultdict(list)
    for event in line_events:
        crossings_by_frame[event["frame_index"]].append(event)
    for row in timeline:
        for event in crossings_by_frame[row["frame_index"]]:
            cumulative[event["event_type"]] += 1
        row["entries"], row["exits"] = cumulative["entry"], cumulative["exit"]
    full_review = all(frame["state"] == "reviewed" for frame in review)
    explicit_ambiguities = sum(frame["state"] == "ambiguous" for frame in review) + sum(bool(a.get("uncertainty")) for a in document["annotations"])
    return {"entries": len([event for event in line_events if event["event_type"] == "entry"]),
            "exits": len([event for event in line_events if event["event_type"] == "exit"]),
            "reviewed_entries": sum(event["confirmed"] and event["event_type"] == "entry" for event in line_events),
            "reviewed_exits": sum(event["confirmed"] and event["event_type"] == "exit" for event in line_events),
            "visitors": None, "live_occupancy": None,
            "unavailable_reason": "Visible track IDs and line crossings do not establish unique customers or initial/live store occupancy.",
            "peak_visible": max(row["visible"] for row in timeline), "mean_span": mean([visit["dwell_seconds"] for visit in visits]),
            "reviewed_dwell": mean([visit["dwell_seconds"] for visit in visits if visit["confirmed"]]),
            "frame_count": len(timestamps), "reviewed_frames": sum(frame["state"] != "unreviewed" for frame in review),
            "fully_reviewed_frames": sum(frame["state"] == "reviewed" for frame in review), "full_review": full_review,
            "corrected_detections": sum(a["origin"] != "raw_model" for a in document["annotations"]) + document.get("deleted_detections", 0),
            "corrected_identity_assignments": sum(bool(a.get("identity_corrected")) for a in document["annotations"]),
            "unconfirmed_identities": sum(not document["identities"][identity]["confirmed"] for identity in identity_annotations),
            "explicit_ambiguities": explicit_ambiguities, "remaining_ambiguities_assessed": full_review,
            "stale_event_reviews": stale, "incomplete_zone_events": incomplete, "suppressed_duplicate_events": duplicate_events,
            "events": events, "visits": visits, "timeline": timeline, "traffic": traffic, "zones": zone_stats,
            "tracks": [{**stat, **document["identities"][identity]} for identity, stat in track_stats.items()]}
