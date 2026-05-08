import json
from urllib.parse import urlparse
from pathlib import Path


def extract_unique_domains(jsonl_path: str) -> set:
    """Extract unique domains from the link field in a JSONL file."""
    domains = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                link = data.get("meta_info", {}).get("link", "")

                if link:
                    parsed = urlparse(link)
                    if parsed.netloc:
                        domains.add(parsed.netloc)
            except json.JSONDecodeError:
                continue

    return domains


if __name__ == "__main__":
    jsonl_file = Path(__file__).parent.parent / "wenetspeech_yue_meta.jsonl"
    domains = extract_unique_domains(str(jsonl_file))

    print(f"Found {len(domains)} unique domains:\n")
    for domain in sorted(domains):
        print(domain)
