# self-guided-learning

## Running `codebase.py`

Basic CLI (required arguments):

```bash
python codebase.py --seed <INT> --mode <MODE>
```

Parameters:

- `--seed`: (int) Random seed for deterministic runs. Example: `0`.
- `--mode`: (str) Use `consensus` to enable the latent-consensus similarity loss (internally sets `lambda_sim=1.0`); any other value disables consensus (sets `lambda_sim=0.0`).

Examples:

```bash
# Run with consensus loss enabled
python codebase.py --seed 0 --mode consensus

# Run without consensus loss
python codebase.py --seed 1 --mode baseline
```

Outputs:

- Per-seed result file: `Shallow_ASPL_FL_Oxford_<mode>/seed_<N>.txt` (accuracy & F1).
- Trained model checkpoint saved as `Shallow_ASPL_FL_Oxford_<mode>/Shallow_ASPL_FL_Oxford_<mode>_seed=<seed>.pth`.
- Latent vectors (if generated) under `Latent_vectors/<seed>/`.

Notes:

- Ensure dataset files are placed under `datasets/oxford_pets/` (images + annotations) or adjust paths in `codebase.py`.
- GPU recommended; script will use CPU if CUDA is unavailable.

