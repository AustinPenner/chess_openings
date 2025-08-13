import os
import json
import hashlib
from typing import List, Optional, Tuple

import chess
import pandas as pd


def download_data(data_dir: str = "data") -> None:
    """Download the Lichess openings dataset and save as Parquet and CSV.
    """
    os.makedirs(data_dir, exist_ok=True)
    # Import here so users who only want FEN conversion don't need datasets/pandas installed
    from datasets import load_dataset

    ds = load_dataset("Lichess/chess-openings", split="train")
    df = ds.remove_columns("img").to_pandas()
    df.to_parquet(os.path.join(data_dir, "chess_openings.parquet"))
    df.to_csv(os.path.join(data_dir, "chess_openings.csv"), index=False)


def epd_sequence_from_uci(uci_moves: str) -> List[str]:
    """Return EPDs after each move in a space-separated UCI sequence.

    Example: "e2e4 e7e5 g1f3 b8c6 f1b5"
    """
    board = chess.Board()
    epds: List[str] = []
    tokens = [t for t in uci_moves.strip().split() if t]
    for token in tokens:
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError(
                f"Illegal UCI move '{token}' for position: {board.epd()}"
            )
        board.push(move)
        epds.append(board.epd())
    return epds


def compute_epd_sequence(row: pd.Series) -> Tuple[List[str], List[str]]:
    """Compute EPD and UCI sequences for a DataFrame row using `uci` only.

    Returns:
        Tuple of (epd_list, uci_list)
    """
    if "uci" not in row or not isinstance(row["uci"], str) or not row["uci"].strip():
        return [], []
    uci_value: str = row["uci"].strip()
    epds = epd_sequence_from_uci(uci_value)
    ucis = [t for t in uci_value.split() if t]
    return epds, ucis


def blake2_epd_hash(epd: str) -> str:
    """Return a compact hex hash for an EPD string for quick lookup."""
    return hashlib.blake2b(epd.encode("utf-8"), digest_size=16).hexdigest()


def build_lookup() -> pd.DataFrame:
    """Load source dataset, explode by halfmoves, compute unique EPD lookup.

    Output columns: ['evo-volume','eco','name','pgn','uci','epd','epdhash']
    """
    data_dir = os.path.join("data")
    source_parquet = os.path.join(data_dir, "chess_openings.parquet")
    source_csv = os.path.join(data_dir, "chess_openings.csv")

    if os.path.exists(source_parquet):
        df = pd.read_parquet(source_parquet)
    elif os.path.exists(source_csv):
        df = pd.read_csv(source_csv)
    else:
        raise FileNotFoundError(
            f"Could not find source dataset at '{source_parquet}' or '{source_csv}'."
        )

    # Explode to one row per halfmove with associated EPD
    exploded_rows: List[dict] = []
    for _, row in df.iterrows():
        epds, ucis = compute_epd_sequence(row)
        if not epds:
            continue
        # Use provided UCI sequence directly
        uci_sequence_str = row.get("uci")

        for idx, epd in enumerate(epds, start=1):
            exploded_rows.append(
                {
                    "evo-volume": row.get("evo-volume"),
                    "eco": row.get("eco"),
                    "name": row.get("name"),
                    "pgn": row.get("pgn"),
                    "uci": uci_sequence_str,
                    "epd": epd,
                    "halfmoves": idx,
                }
            )

    exploded_df = pd.DataFrame(exploded_rows)

    # Order by number of halfmoves ascending, then drop duplicates by EPD keeping first
    exploded_df = exploded_df.sort_values(["halfmoves"]).reset_index(drop=True)
    dedup_df = exploded_df.drop_duplicates(subset=["epd"], keep="first").copy()

    # Compute epdhash
    dedup_df["epdhash"] = dedup_df["epd"].map(blake2_epd_hash)

    # Final column selection and ordering
    final_df = dedup_df[["evo-volume", "eco", "name", "pgn", "uci", "epd", "epdhash"]]
    return final_df


def write_outputs(df: pd.DataFrame) -> None:
    """Write lookup DataFrame to parquet, csv, and json in data/ directory."""
    data_dir = os.path.join("data")
    os.makedirs(data_dir, exist_ok=True)
    parquet_path = os.path.join(data_dir, "chess_openings_lookup.parquet")
    csv_path = os.path.join(data_dir, "chess_openings_lookup.csv")
    json_path = os.path.join(data_dir, "chess_openings_lookup.json")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False)


if __name__ == "__main__":
    # One-off build: explode by halfmoves, unique by EPD, and write lookup outputs
    lookup_df = build_lookup()
    write_outputs(lookup_df)
    # Also print JSON to stdout for convenience
    print(json.dumps(lookup_df.to_dict(orient="records"), ensure_ascii=False))