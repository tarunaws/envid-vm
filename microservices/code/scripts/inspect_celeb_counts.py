import json
import sys
import urllib.request

VIDEO_ID = sys.argv[1] if len(sys.argv) > 1 else ""
if not VIDEO_ID:
    raise SystemExit("Usage: inspect_celeb_counts.py <video_id> [base_url]")

BASE_URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5016"
url = f"{BASE_URL.rstrip('/')}/video/{VIDEO_ID}/metadata-json?category=combined"

with urllib.request.urlopen(url, timeout=30) as r:
    j = json.loads(r.read().decode("utf-8"))

cats = j.get("categories") or {}
ct = cats.get("celebrity_table") or cats.get("celebrity_detection") or {}
celebs = ct.get("celebrities") or []

wanted = ["Kunal Khemu", "Tanuj", "Rahul Dev", "Asha Negi", "Vijay Raj"]
want = {w.lower() for w in wanted}

print("url", url)
print("celebs entries", len(celebs))

print("\nAll celebrity names (with occurrences):")
for c in celebs:
    nm = (c.get("name") or "").strip()
    if not nm:
        continue
    occ = c.get("occurrences") or c.get("occurrence_count") or c.get("count")
    print(f"- {nm}: {occ}")

for c in celebs:
    nm = (c.get("name") or "").strip()
    if nm.lower() not in want:
        continue

    occ = c.get("occurrences") or c.get("occurrence_count") or c.get("count")
    segs = c.get("segments") or []
    times = c.get("timestamps") or c.get("timecodes") or c.get("occurrence_timestamps") or []
    ts_s = c.get("timestamps_seconds") or []
    ts_ms = c.get("timestamps_ms") or []
    dets = c.get("detections") or []

    print("\nNAME", nm)
    print("  occurrences field:", occ)
    print("  segments:", len(segs))
    print("  timestamps:", len(times))
    print("  timestamps_seconds:", len(ts_s))
    print("  timestamps_ms:", len(ts_ms))
    print("  detections:", len(dets))

    if isinstance(segs, list) and segs and isinstance(segs[0], dict):
        first = segs[0]
        show = {k: first.get(k) for k in ("start_ms", "end_ms", "start_tc", "end_tc", "start_seconds", "end_seconds") if k in first}
        print("  first segment:", show)

    if isinstance(times, list) and times:
        print("  first 10 times:", times[:10])

    if isinstance(ts_s, list) and ts_s:
        print("  first 10 timestamps_seconds:", ts_s[:10])

    if isinstance(ts_ms, list) and ts_ms:
        print("  first 10 timestamps_ms:", ts_ms[:10])
