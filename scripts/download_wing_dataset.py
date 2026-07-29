"""Download the published wing-training CSV and its configuration from Hugging Face."""

from pathlib import Path
import shutil

from huggingface_hub import snapshot_download

DATASET_REPOSITORY = "jpfenniger/Aero-Struct-Wing"
REQUIRED_FILES = ("combined_wings_data_clipped.csv", "config.yaml")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    destination = project_root / "data" / "processed" / "wings"
    destination.mkdir(parents=True, exist_ok=True)

    missing_files = [name for name in REQUIRED_FILES if not (destination / name).is_file()]
    if not missing_files:
        print(f"Wing dataset is already available at {destination}")
        return

    print(f"Downloading {DATASET_REPOSITORY} ...")
    snapshot = Path(snapshot_download(repo_id=DATASET_REPOSITORY, repo_type="dataset"))

    for file_name in missing_files:
        matches = [path for path in snapshot.rglob(file_name) if path.is_file()]
        if not matches:
            raise FileNotFoundError(
                f"The dataset repository did not contain {file_name}. "
                f"See https://huggingface.co/datasets/{DATASET_REPOSITORY}."
            )
        target = destination / file_name
        shutil.copy2(matches[0], target)
        print(f"Saved {target}")


if __name__ == "__main__":
    main()
