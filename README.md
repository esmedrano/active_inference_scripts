# AIF

To be clear I basically copy pasted this from the book Fundamentals of Active Inference by Sanjeev Namjoshi. It provides the formula derivations and sudo code for this implementation. I did the derivations though. Most of this is coded by hand but I did get help from Gemini for the graph function and a few bugs.

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
