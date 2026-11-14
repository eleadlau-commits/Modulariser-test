# Run offline on Windows (no GPU)

Needs Python 3.10–3.12 and about 3 GB of free RAM. 16 GB laptops are fine.

1. Install Python from python.org (tick "Add Python to PATH").
2. Open PowerShell and run:
   ```
   git clone https://github.com/eleadlau-commits/Modulariser-test
   cd Modulariser-test
   py -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   ```
   (No git? Download the repo as a ZIP from GitHub and unzip it instead.)
3. Run once while online, so it downloads the model (~1.9 GB):
   ```
   python local\run_local.py examples\bakery.txt
   ```
4. From then on it works with no internet. Cut your own notes with
   `python local\run_local.py my_notes.txt` (add `--json` for JSON only).

Each later session: `cd Modulariser-test` then `.venv\Scripts\activate` first.
