## Chess Openings Utilities (minimal)

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Download dataset

```bash
python3 main.py download
```

Outputs:
- `data/chess_openings.parquet`
- `data/chess_openings.csv`

### Convert PGN/ UCI to FENs

Examples:

```bash
python3 main.py fen --pgn "1. e4 e5 2. Nf3 Nc6 3. Bb5"
```

```bash
python3 main.py fen --uci "e2e4 e7e5 g1f3 b8c6 f1b5"
```

Each prints one FEN per move.



