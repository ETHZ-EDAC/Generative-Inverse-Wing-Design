# Conditional Wing VAE, release 0.1

`wing_cvae_full_data_v0.1.ckpt` is the selected full-data conditional wing VAE checkpoint distributed with this release.

It is a PyTorch Lightning checkpoint trained on the published aero-structural wing dataset.

The selected run reached validation loss `4.0751` and test loss `4.0826`.

The checkpoint is 1.22 MB.

## Requirements

The cVAE setup requires `data/processed/wings/combined_wings_data_clipped.csv` and `data/processed/wings/config.yaml` to reconstruct the feature scaler and wing configuration.

Download these required files with:

```bash
uv run python scripts/download_wing_dataset.py
```

The checkpoint is intended for use within the cVAE training notebook and model workflow supplied in this repository.

