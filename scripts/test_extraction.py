"""
Quick test to extract first 100 entries and print summary.
No downloads - just shows what would be downloaded.
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_JSONL = Path(__file__).parent.parent / "wenetspeech_yue_meta.jsonl"
COOKIES_DIR = Path(__file__).parent.parent / "scripts" / "cookies"
OUTPUT_DIR = Path(__file__).parent.parent / "data"

PLATFORM_CONFIG = {
    "douyin": {"domains": ["douyin.com"]},
    "bilibili": {"domains": ["bilibili.com", "space.bilibili.com"]},
    "ixigua": {"domains": ["ixigua.com"]},
    "acfun": {"domains": ["acfun.cn"]},
}


def detect_platform(url: str) -> (str
                                  | None):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    for platform, config in PLATFORM_CONFIG.items():
        if any(d in domain for d in config["domains"]):
            return platform
    return None


def parse_timestamp(ts: str) -> tuple[float, float]:
    parts = ts.split("_")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return 0.0, 0.0
    return 0.0, 0.0


def main():
    print(f"Reading {DEFAULT_JSONL.name}...")

    entries = []
    completed = set()

    manifest_file = OUTPUT_DIR / "download_manifest.json"
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            completed = set(manifest.get("completed", []))

    print(f"Already completed: {len(completed)}")

    with open(DEFAULT_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break

            try:
                data = json.loads(line.strip())
                key = data.get("key")

                if key in completed:
                    continue

                link = data.get("meta_info", {}).get("link", "")
                time_stamp = data.get("meta_info", {}).get("time_stamp", "")

                if not link or not time_stamp:
                    continue

                platform = detect_platform(link)
                if not platform:
                    continue

                start_sec, end_sec = parse_timestamp(time_stamp)
                duration = data.get("duration", end_sec - start_sec)

                entries.append(
                    {
                        "key": key,
                        "platform": platform,
                        "duration": duration,
                        "url": link,
                    }
                )

            except json.JSONDecodeError:
                continue

    print(f"\nTotal entries with valid links (first 10k lines): {len(entries)}")

    platforms = {}
    total_duration = 0
    for e in entries:
        p = e["platform"]
        platforms[p] = platforms.get(p, 0) + 1
        total_duration += e["duration"]

    print("\nPlatform breakdown:")
    for p, count in sorted(platforms.items()):
        print(f"  {p}: {count}")

    print(f"\nTotal audio duration: {total_duration / 3600:.1f} hours")
    print(
        f"Estimated storage: ~{total_duration * 16000 * 2 / 1e9:.1f} GB (16kHz mono WAV)"
    )

    print(f"\n--- Cookie files needed ---")
    print(f"Location: {COOKIES_DIR}")

    has_cookies = False
    for platform in PLATFORM_CONFIG:
        cookie_file = COOKIES_DIR / f"{platform}.txt"
        if cookie_file.exists():
            print(f"  {platform}.txt: OK")
            has_cookies = True
        else:
            print(f"  {platform}.txt: MISSING")

    if not has_cookies:
        print("\n[!] No cookie files found!")
        print("\nTo export cookies from your browser:")
        print("  1. Install 'Get cookies.txt LOCALLY' extension (Chrome/Firefox)")
        print("  2. Visit each domain and export cookies")
        print(f"  3. Save to {COOKIES_DIR}/")

    print("\n--- Next steps ---")
    print("  1. Export cookie files to cookies/ directory")
    print("  2. Install FFmpeg: winget install ffmpeg")
    print("  3. Run: python scripts/download_audio.py --limit 1000")


if __name__ == "__main__":
    main()
