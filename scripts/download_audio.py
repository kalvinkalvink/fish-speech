"""
Download audio from WenetSpeech-Yue dataset.

Usage:
    # Test run with 100 entries
    python scripts/download_audio.py --test

    # Full download (all ~3M entries with links)
    python scripts/download_audio.py

    # Custom limit
    python scripts/download_audio.py --limit 10000

    # Resume from manifest
    python scripts/download_audio.py --resume

    # Disable video (audio only)
    python scripts/download_audio.py --audio-only
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_JSONL = Path(__file__).parent.parent / "wenetspeech_yue_meta.jsonl"
COOKIES_DIR = Path(__file__).parent.parent / "cookies"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
MANIFEST_FILE = OUTPUT_DIR / "download_manifest.json"

MAX_CONCURRENT = 10
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

PLATFORM_CONFIG = {
    "douyin": {
        "domains": ["douyin.com", "www.douyin.com"],
        "format": "bestaudio/best",
        "extractor": "Douyin",
    },
    "bilibili": {
        "domains": ["bilibili.com", "www.bilibili.com", "space.bilibili.com"],
        "format": "bestaudio/best",
        "extractor": "Bilibili",
    },
    "ixigua": {
        "domains": ["ixigua.com", "www.ixigua.com"],
        "format": "bestaudio/best",
        "extractor": "Ixigua",
    },
    "acfun": {
        "domains": ["acfun.cn", "www.acfun.cn"],
        "format": "bestaudio/best",
        "extractor": "AcFun",
    },
}


def detect_platform(url: str) -> str | None:
    """Detect platform from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    for platform, config in PLATFORM_CONFIG.items():
        if any(d in domain for d in config["domains"]):
            return platform
    return None


def parse_timestamp(ts: str) -> tuple[float, float]:
    """Parse time_stamp string like '620.270_624.350' to (start, end) in seconds."""
    parts = ts.split("_")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return 0.0, 0.0
    return 0.0, 0.0


def load_manifest() -> dict:
    """Load download manifest."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "completed": [],
        "failed": {},
        "pending": 0,
        "total": 0,
        "last_updated": None,
    }


def save_manifest(manifest: dict):
    """Save download manifest."""
    manifest["last_updated"] = datetime.now().isoformat()
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


async def extract_entries(
    jsonl_path: Path,
    max_entries: int | None = None,
    skip_completed: bool = True,
) -> list[dict]:
    """Extract entries from JSONL file."""
    entries = []
    manifest = load_manifest() if skip_completed else {"completed": [], "failed": {}}
    completed_keys = set(manifest["completed"])

    print(f"Reading {jsonl_path.name}...")
    print(
        f"(Skipping {len(completed_keys)} already completed)" if skip_completed else ""
    )

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Reading")):
            if max_entries and i >= max_entries:
                break

            try:
                data = json.loads(line.strip())
                key = data.get("key")

                if skip_completed and key in completed_keys:
                    continue

                link = data.get("meta_info", {}).get("link", "")
                time_stamp = data.get("meta_info", {}).get("time_stamp", "")

                if not link or not time_stamp:
                    continue

                platform = detect_platform(link)
                if not platform:
                    continue

                start_sec, end_sec = parse_timestamp(time_stamp)

                entries.append(
                    {
                        "key": key,
                        "link": link,
                        "time_stamp": time_stamp,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "duration": data.get("duration", end_sec - start_sec),
                        "platform": platform,
                    }
                )

            except json.JSONDecodeError:
                continue

    return entries


async def download_single_video(
    entry: dict,
    cookies_dir: Path,
    keep_video: bool = True,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Download single video using yt-dlp."""
    url = entry["link"]
    key = entry["key"]
    platform = entry["platform"]
    start_sec = entry["start_sec"]
    end_sec = entry["end_sec"]

    cookies_file = cookies_dir / f"{platform}.txt"
    cookies_arg = str(cookies_file) if cookies_file.exists() else None

    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(temp_dir / "%(id)s.%(ext)s")

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--quiet",
        "--no-playlist",
        "--no-warnings",
    ]

    if cookies_arg:
        cmd.extend(["--cookies", cookies_arg])

    if keep_video:
        cmd.extend(
            [
                "-f",
                "bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/bestvideo+bestaudio/best",
            ]
        )
    else:
        cmd.extend(["-f", PLATFORM_CONFIG[platform]["format"]])

    cmd.extend(["-o", output_template, url])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            if verbose:
                print(f"yt-dlp error: {result.stderr}")
            return False, f"yt-dlp failed: {result.returncode}"

        temp_files = (
            list(temp_dir.glob("*.mp4"))
            + list(temp_dir.glob("*.m4a"))
            + list(temp_dir.glob("*.webm"))
        )

        if not temp_files:
            return False, "No downloaded file found"

        input_file = temp_files[0]

        output_audio = OUTPUT_DIR / "audio" / f"{key}.wav"
        output_audio.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-ss",
            str(start_sec),
            "-to",
            str(end_sec),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(output_audio),
        ]

        try:
            subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                timeout=60,
                check=True,
            )
        except subprocess.TimeoutExpired:
            input_file.unlink()
            return False, "FFmpeg timeout"
        except subprocess.CalledProcessError as e:
            input_file.unlink()
            return False, f"FFmpeg error: {e}"

        input_file.unlink()

        if not keep_video:
            return True, "Downloaded"

        output_video = OUTPUT_DIR / "video" / f"{key}.mp4"
        output_video.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(input_file), str(output_video))

        return True, "Downloaded"

    except subprocess.TimeoutExpired:
        return False, "Download timeout"
    except Exception as e:
        return False, f"Error: {str(e)}"


async def download_batch(
    entries: list[dict],
    cookies_dir: Path,
    keep_video: bool = True,
    max_concurrent: int = MAX_CONCURRENT,
    verbose: bool = False,
) -> tuple[int, int]:
    """Download batch with parallelism."""
    manifest = load_manifest()
    completed = manifest["completed"]
    failed = manifest["failed"]

    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_with_semaphore(entry: dict):
        async with semaphore:
            key = entry["key"]

            for attempt in range(RETRY_ATTEMPTS):
                success, status = await download_single_video(
                    entry, cookies_dir, keep_video, verbose
                )

                if success:
                    completed.append(key)
                    return 1, 0

                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY)

            failed[key] = status
            return 0, 1

    print(f"\nDownloading {len(entries)} entries with {max_concurrent} concurrent...")

    tasks = []
    pbar = tqdm(total=len(entries), desc="Downloading")

    for entry in entries:
        task = asyncio.create_task(download_with_semaphore(entry))
        tasks.append((task, pbar))

    success_count = 0
    fail_count = 0

    for task, pbar in tasks:
        s, f = await task
        success_count += s
        fail_count += f
        pbar.update(1)

        if (len(completed) + len(failed)) % 100 == 0:
            manifest = load_manifest()
            save_manifest(manifest)

    pbar.close()

    manifest["completed"] = completed
    manifest["failed"] = failed
    manifest["total"] = len(completed) + len(failed)
    save_manifest(manifest)

    return success_count, fail_count


async def run_test(max_entries: int = 100):
    """Run test with limited entries."""
    print(f"=== Test Run: {max_entries} entries ===")

    entries = await extract_entries(DEFAULT_JSONL, max_entries=max_entries)
    print(f"Extracted: {len(entries)} entries")

    if not entries:
        print("No entries found!")
        return

    platforms = {}
    for e in entries:
        p = e["platform"]
        platforms[p] = platforms.get(p, 0) + 1

    print("Platform breakdown:")
    for p, c in platforms.items():
        print(f"  {p}: {c}")

    print(f"\nCookies needed in: {COOKIES_DIR}")
    for platform in PLATFORM_CONFIG:
        cookie_file = COOKIES_DIR / f"{platform}.txt"
        status = "FOUND" if cookie_file.exists() else "MISSING"
        print(f"  {platform}.txt: {status}")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Manifest file: {MANIFEST_FILE}")

    print("\n=== Proceeding with test download ===")

    success, failed = await download_batch(
        entries,
        COOKIES_DIR,
        keep_video=True,
    )

    print(f"\n=== Results ===")
    print(f"Success: {success}")
    print(f"Failed: {failed}")


async def run_full(limit: int | None = None, resume: bool = False):
    """Run full download."""
    print("=== Full Download ===")

    entries = await extract_entries(
        DEFAULT_JSONL, max_entries=limit, skip_completed=resume
    )
    print(f"Pending entries: {len(entries)}")

    if not entries:
        print("No entries to download!")
        return

    platforms = {}
    for e in entries:
        p = e["platform"]
        platforms[p] = platforms.get(p, 0) + 1

    print("Platform breakdown:")
    for p, c in platforms.items():
        print(f"  {p}: {c}")

    print(f"\nStarting download...")
    success, failed = await download_batch(
        entries,
        COOKIES_DIR,
        keep_video=True,
    )

    print(f"\n=== Results ===")
    print(f"Success: {success}")
    print(f"Failed: {failed}")


def show_status():
    """Show current status."""
    manifest = load_manifest()

    print("=== Download Status ===")
    total_processed = len(manifest["completed"]) + len(manifest["failed"])
    print(f"Total processed: {total_processed}")
    print(f"Completed: {len(manifest['completed'])}")
    print(f"Failed: {len(manifest['failed'])}")
    print(f"Last updated: {manifest.get('last_updated', 'Never')}")

    if manifest["failed"]:
        print("\nRecent failures:")
        for key, err in list(manifest["failed"].items())[-5:]:
            print(f"  {key}: {err}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download audio from WenetSpeech-Yue")
    parser.add_argument("--test", action="store_true", help="Test run with 100 entries")
    parser.add_argument(
        "--limit", type=int, default=1000, help="Limit number of entries"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from manifest")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument(
        "--audio-only", action="store_true", help="Audio only (no video)"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help="Max concurrent downloads",
    )
    parser.add_argument("--all", action="store_true", help="Download all entries")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.test:
        asyncio.run(run_test(100))
    elif args.all:
        asyncio.run(run_full(None, args.resume))
    else:
        asyncio.run(run_full(args.limit, args.resume))


if __name__ == "__main__":
    asyncio.run(main())
