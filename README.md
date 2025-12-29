## Chess Openings Utilities (minimal)

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Download dataset

To download the Lichess openings dataset, open a Python shell and run:

```python
from main import download_data
download_data()
```

Outputs:
- `data/chess_openings.parquet`
- `data/chess_openings.csv`

### Build lookup table

Once the dataset is downloaded, run:

```bash
python3 main.py
```

This will:
1. Load the source dataset from `data/`
2. Explode each opening into one row per halfmove
3. Compute EPD (board position) for each move
4. Deduplicate by EPD, keeping the shortest path to each position
5. Output the lookup table in multiple formats

Outputs:
- `data/chess_openings_lookup.parquet`
- `data/chess_openings_lookup.csv`
- `data/chess_openings_lookup.json`

The JSON is also printed to stdout.

