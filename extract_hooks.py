import os
import json

HOOK_LIBRARY = []

# You can change this!
MIN_VIEWS = 100000  # Only include videos with >= 100k views
INCLUDE_TYPES = ["gold", "public"]  # Or ["gold"] for just gold winners
STYLES = ["conspiracy", "personal story"]  # Add any you want, or remove filter

def extract_hook(transcript):
    # Grab first 2-3 sentences or 8-12 seconds (approx)
    if not transcript:
        return ""
    # Split by sentences or timestamps if present
    sentences = transcript.split('. ')
    return '. '.join(sentences[:2]).strip() + ("." if sentences else "")

for filename in os.listdir("."):
    if filename.startswith("library_") and filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
            # Filter by type
            type_match = (
                ("gold" in INCLUDE_TYPES and filename.startswith("library_gold_winner_")) or
                ("public" in INCLUDE_TYPES and filename.startswith("library_public_winner_"))
            )
            if not type_match:
                continue
            # Filter by style
            style = data.get("style", "").lower()
            if STYLES and style not in STYLES:
                continue
            # Filter by views
            if data.get("views", 0) < MIN_VIEWS:
                continue
            # Grab transcript from deconstruction
            transcript = None
            try:
                transcript = (
                    data["deconstruction"].get("transcript")
                    or data["deconstruction"]["deconstruction"].get("transcript")
                )
            except Exception:
                pass
            if not transcript:
                continue
            hook = extract_hook(transcript)
            HOOK_LIBRARY.append({
                "video_url": data.get("video_url"),
                "style": style,
                "views": data.get("views"),
                "hook": hook
            })

# Save out for inspection
with open("hook_library.json", "w") as f:
    json.dump(HOOK_LIBRARY, f, indent=2)

print(f"Extracted {len(HOOK_LIBRARY)} hooks from your video library.")
