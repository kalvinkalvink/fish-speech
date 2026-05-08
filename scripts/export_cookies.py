import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
import aiohttp
from tqdm.asyncio import tqdm

COOKIE_DOMAINS = [
    "douyin.com",
    "bilibili.com",
    "ixigua.com",
    "acfun.cn",
]


def detect_platform(url: str) -> str | None:
    """Detect platform from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if "douyin.com" in domain:
        return "douyin"
    elif "bilibili.com" in domain or "bilibili.com" in domain:
        return "bilibili"
    elif "ixigua.com" in domain:
        return "ixigua"
    elif "acfun.cn" in domain:
        return "acfun"
    return None


async def download_with_ytdlp(
    url: str,
    output_path: str,
    cookies_file: str | None = None,
    platform: str = "douyin",
) -> bool:
    """Download video using yt-dlp."""
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-download",
        "--no-playlist",
        "--flat-playlist",
        "--dump-json",
    ]

    if cookies_file and Path(cookies_file).exists():
        cmd.extend(["--cookies", cookies_file])

    cmd.append(url)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode == 0 and stdout:
            data = json.loads(stdout.decode())
            return True
    except Exception as e:
        pass

    return False


async def check_video_available(
    url: str,
    cookies_dir: Path,
) -> tuple[bool, str]:
    """Check if video is available for download."""
    platform = detect_platform(url)
    if not platform:
        return False, "Unknown platform"

    cookies_file = cookies_dir / f"{platform}.txt"

    available = await download_with_ytdlp(
        url,
        str(cookies_file.resolve()),
        str(cookies_file.resolve()) if cookies_file.exists() else None,
        platform,
    )

    if available:
        return True, "Available"
    else:
        return False, "Unavailable or requires authentication"


async def extract_metadata_batch(
    jsonl_path: str,
    max_entries: int = 100,
) -> list[dict]:
    """Extract metadata from JSONL file."""
    entries = []

    async with aiofiles.open(jsonl_path, "r", encoding="utf-8") as f:
        async for line in f:
            if len(entries) >= max_entries:
                break

            try:
                data = json.loads(line.strip())
                link = data.get("meta_info", {}).get("link", "")
                time_stamp = data.get("meta_info", {}).get("time_stamp", "")
                duration = data.get("duration", 0)

                if link and time_stamp:
                    platform = detect_platform(link)
                    if platform:
                        entries.append(
                            {
                                "key": data.get("key"),
                                "link": link,
                                "time_stamp": time_stamp,
                                "duration": duration,
                                "platform": platform,
                            }
                        )
            except json.JSONDecodeError:
                continue

    return entries


def parse_timestamp(ts: str) -> tuple[float, float]:
    """Parse time_stamp string like '620.270_624.350' to (start, end) in seconds."""
    parts = ts.split("_")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return 0.0, 0.0
    return 0.0, 0.0


async def main():
    """Main entry point."""
    # Test extraction first
    jsonl_path = Path(__file__).parent.parent / "wenetspeech_yue_meta.jsonl"
    cookies_dir = Path(__file__).parent.parent / "cookies"

    print("Extracting sample entries...")
    entries = await extract_metadata_batch(str(jsonl_path), 10)

    print(f"\nFound {len(entries)} entries with links in sample (10)")
    print("\nPlatform breakdown:")

    platforms = {}
    for entry in entries:
        p = entry["platform"]
        platforms[p] = platforms.get(p, 0) + 1

    for p, count in platforms.items():
        print(f"  {p}: {count}")

    print(f"\nSample entries:")
    for i, entry in enumerate(entries[:3]):
        start, end = parse_timestamp(entry["time_stamp"])
        print(f"  {i + 1}. {entry['key']}")
        print(f"     URL: {entry['link']}")
        print(f"     Platform: {entry['platform']}")
        print(
            f"     Time: {start:.2f}s - {end:.2f}s (duration: {entry['duration']:.2f}s)"
        )

    print(f"\n--- Testing video availability check ---")

    # Check first entry availability
    if entries:
        entry = entries[0]
        available, status = await check_video_available(entry["link"], cookies_dir)
        print(f"\nChecking: {entry['link']}")
        print(f"Available: {available}, Status: {status}")

    print("\nCookie files needed in:")
    print(f"  {cookies_dir.resolve()}")
    for domain in COOKIE_DOMAINS:
        print(f"    - {domain}.txt (export from browser)")


if __name__ == "__main__":
    asyncio.run(main())
