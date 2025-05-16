import requests
import os
import json
from tqdm import tqdm

def list_venusx_datasets(user="AI4Protein", prefix="VenusX"):
    """
    get all dataset ids of Hugging Face user with prefix
    """
    url = f"https://huggingface.co/api/datasets?author={user}"
    response = requests.get(url)
    response.raise_for_status()
    datasets = response.json()

    return [ds["id"] for ds in datasets if ds["id"].startswith(f"{user}/{prefix}")]

def fetch_and_save_croissant(dataset_id, output_dir):
    """
    fetch and save single dataset croissant metadata
    """
    url = f"https://huggingface.co/api/datasets/{dataset_id}/croissant"
    try:
        response = requests.get(url)
        response.raise_for_status()
        metadata = response.json()

        dataset_name = dataset_id.split("/")[-1]
        file_path = os.path.join(output_dir, f"{dataset_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        print(f"⚠️ failed: {dataset_id} — {e}")
        return False

def download_all_croissant_metadata(output_dir="croissant_files"):
    """
    get all croissant metadata from VenusX
    """
    os.makedirs(output_dir, exist_ok=True)

    print("📡 getting VenusX dataset list...")
    dataset_ids = list_venusx_datasets()
    print(f"🔍 find {len(dataset_ids)} datasets.")

    for dataset_id in tqdm(dataset_ids, desc="📥 downloading croissant metadata"):
        fetch_and_save_croissant(dataset_id, output_dir)

    print(f"\n✅ all croissant metadata saved to {output_dir}")

if __name__ == "__main__":
    download_all_croissant_metadata()
