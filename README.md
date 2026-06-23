# AIF

This repository contains a Python simulation of generalized filtering using a simple active inference model.

## Setup

1. Create a Python virtual environment if you do not already have one:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

The current dependencies are:

- `matplotlib==3.11.0`
- `numpy==2.4.6`
- `scipy==1.17.1`

## Run

```bash
python generalised_filtering.py
```

## Notes

- The dependency file `requirements.txt` was generated from the project virtual environment.
- `.venv/` is ignored in Git so the repository stays lightweight.
