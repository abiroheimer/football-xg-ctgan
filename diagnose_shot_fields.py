"""
diagnose_shot_fields.py — One-off diagnostic: print the raw tags/subEventId
structure of a handful of real shot events, to find out why body_part()/
play_pat() are defaulting to "Right Foot"/"Open Play" for every shot.

Run from "final codes" with venv active:
    python diagnose_shot_fields.py
"""
import json
import os

from config import WYSCOUT_DIR

ev_path = os.path.join(WYSCOUT_DIR, "events_England.json")
print(f"Reading {ev_path} ...\n")

with open(ev_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

print(f"Total events in file: {len(raw):,}")
print(f"Type of raw: {type(raw)}")
print(f"Keys of first event: {list(raw[0].keys())}\n")

# Find the first several shot events (eventId == 10)
shot_count = 0
for e in raw:
    if int(e.get("eventId", -1)) == 10:
        shot_count += 1
        print("=" * 60)
        print(f"SHOT #{shot_count}")
        print("=" * 60)
        print(f"  eventId      : {e.get('eventId')}  (type={type(e.get('eventId')).__name__})")
        print(f"  subEventId   : {e.get('subEventId')}  (type={type(e.get('subEventId')).__name__})")
        print(f"  subEventName : {e.get('subEventName')}")
        print(f"  tags raw     : {e.get('tags')}")
        print(f"  tags type    : {type(e.get('tags')).__name__}")
        if isinstance(e.get("tags"), list) and e.get("tags"):
            print(f"  tags[0] type : {type(e['tags'][0]).__name__}  value: {e['tags'][0]}")
        print()
        if shot_count >= 8:
            break

if shot_count == 0:
    print("⚠ No events with eventId==10 found in this file at all — check the field name/value.")
