import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
REPO_ID = "ASLP-lab/WenetSpeech-Yue"
FILENAME = "wenetspeech_yue_meta.jsonl"
OUTPUT_DIR = Path("data")  # Where you want the folders to go
# ---------------------

def setup_dataset():
    # 1. Download the metadata file using huggingface_hub
    print(f"Downloading {FILENAME} from {REPO_ID}...")
    meta_path = hf_hub_download(
        repo_id=REPO_ID, 
        filename=FILENAME, 
        repo_type="dataset",
        local_dir="."
    )

    print("Processing metadata and creating .lab files...")
    
    # 2. Process the JSONL file
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Extract relevant fields
            # Note: WenetSpeech-Yue uses 'utt_id' for filenames and 'speaker' for SPK folders
            utt_id = data.get("key")  # e.g., "xg0054364_..."
            text = data.get("rover_result")  # The Cantonese transcription
            speaker_id = data.get("meta_info", {}).get("speaker_id", "UNKNOWN_SPK")

            if not utt_id or not text:
                continue

            # 3. Create Speaker Directory
            spk_dir = OUTPUT_DIR / str(speaker_id)
            spk_dir.mkdir(parents=True, exist_ok=True)

            # 4. Write the .lab file
            lab_path = spk_dir / f"{utt_id}.lab"
            with open(lab_path, 'w', encoding='utf-8') as lab_file:
                lab_file.write(text)

    print(f"Done! Your dataset structure is ready in ./{OUTPUT_DIR}")

if __name__ == "__main__":
    setup_dataset()