# Airfoil VAE, release 0.1

`airfoil_vae_full_data_v0.1.pt` is the release-ready airfoil variational autoencoder.

It learns a two-dimensional latent representation of CST/Kulfan airfoil geometry and the associated aerodynamic quantities used by this project.

The model was trained on the processed IDEALLab airfoil dataset created by `notebooks/airfoils_preprocessing.ipynb`.

The selected baseline run reached validation loss `1.1348` and test loss `1.2159`.

The model file is 392 KB.

## Loading

The native model file contains the architecture, fitted aerodynamic standardization buffers, and training configuration.

```python
from pathlib import Path
from generative_inverse_wing_design.models.VAE.preprocessing_VAE import VAE

model, _ = VAE.load(Path("models/airfoil_vae/airfoil_vae_full_data_v0.1.pt"))
model.eval()
```

