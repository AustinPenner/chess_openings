import os
import json
import hashlib
from typing import Iterable, List, Optional, Tuple

import chess
import pandas as pd
import pyarrow  # noqa: F401  (ensure pyarrow is available for to_parquet)
from io import StringIO
import chess.pgn


def download_data(data_dir: str = "data") -> None:
    """Download the Lichess openings dataset and save as Parquet and CSV.

    This mirrors the snippet provided in the request, with the small addition of
    importing inside the function to keep optional deps minimal for FEN-only use.
    """
    os.makedirs(data_dir, exist_ok=True)
    # Import here so users who only want FEN conversion don't need datasets/pandas installed
    from datasets import load_dataset  # type: ignore
    import pandas as pd  # type: ignore  # noqa: F401  (imported for side-effects/typing)

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


def epd_and_uci_sequence_from_pgn(pgn_moves: str) -> Tuple[List[str], List[str]]:
    """Return (EPD list, UCI list) after each move parsed from a PGN string.

    Uses python-chess PGN parsing to handle SAN, move numbers, and annotations.
    """
    game = chess.pgn.read_game(StringIO(pgn_moves))
    if game is None:
        # Fallback: try naive splitting by SAN tokens if PGN header parsing fails
        board = chess.Board()
        epds: List[str] = []
        ucis: List[str] = []
        tokens = [t for t in pgn_moves.replace("\n", " ").split(" ") if t]
        for token in tokens:
            # Skip move numbers and result markers
            if token.endswith(".") or token.replace(".", "").isdigit() or token in {"1-0", "0-1", "1/2-1/2", "*"}:
                continue
            move = board.parse_san(token)
            if move not in board.legal_moves:
                raise ValueError(f"Illegal SAN move '{token}' for position: {board.epd()}")
            board.push(move)
            epds.append(board.epd())
            ucis.append(move.uci())
        return epds, ucis

    board = game.board()
    epds: List[str] = []
    ucis: List[str] = []
    for move in game.mainline_moves():
        board.push(move)
        epds.append(board.epd())
        ucis.append(move.uci())
    return epds, ucis


def compute_epd_sequence(row: pd.Series) -> Tuple[List[str], List[str]]:
    """Compute EPD and UCI sequences for a DataFrame row using `uci` if present, else `pgn`.

    Returns:
        Tuple of (epd_list, uci_list)
    """
    uci_value: Optional[str] = None
    pgn_value: Optional[str] = None

    # Attempt flexible access to common field names
    if "uci" in row and isinstance(row["uci"], str) and row["uci"].strip():
        uci_value = row["uci"].strip()
    if "pgn" in row and isinstance(row["pgn"], str) and row["pgn"].strip():
        pgn_value = row["pgn"].strip()

    if uci_value:
        epds = epd_sequence_from_uci(uci_value)
        # Build UCI list from tokens to keep it aligned with EPD list
        ucis = [t for t in uci_value.split() if t]
        return epds, ucis
    if pgn_value:
        return epd_and_uci_sequence_from_pgn(pgn_value)

    # No move data available
    return [], []


def first_present_column(df: pd.DataFrame, candidates: Iterable[str], default_name: str) -> str:
    """Return the first candidate column present in df; if none, create a default column.

    Returns the column name that should be used going forward.
    """
    for name in candidates:
        if name in df.columns:
            return name
    # Ensure a default exists for downstream code
    if default_name not in df.columns:
        df[default_name] = pd.NA
    return default_name


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

    # Normalize expected columns, allowing for slight naming differences
    evo_col = first_present_column(df, ["evo-volume", "evo_volume", "evo"], "evo-volume")
    eco_col = first_present_column(df, ["eco"], "eco")
    name_col = first_present_column(df, ["name", "opening_name", "title"], "name")
    pgn_col = first_present_column(df, ["pgn"], "pgn")
    uci_col = first_present_column(df, ["uci"], "uci")

    # Explode to one row per halfmove with associated EPD
    exploded_rows: List[dict] = []
    for _, row in df.iterrows():
        epds, ucis = compute_epd_sequence(row)
        if not epds:
            continue
        # If original UCI is missing but we have UCIs, keep the joined sequence for reference
        uci_sequence_str = row.get(uci_col)
        if not isinstance(uci_sequence_str, str) or not uci_sequence_str.strip():
            uci_sequence_str = " ".join(ucis)

        for idx, epd in enumerate(epds, start=1):
            exploded_rows.append(
                {
                    "evo-volume": row.get(evo_col),
                    "eco": row.get(eco_col),
                    "name": row.get(name_col),
                    "pgn": row.get(pgn_col),
                    "uci": uci_sequence_str,
                    "epd": epd,
                    "halfmoves": idx,
                }
            )

    if not exploded_rows:
        # Nothing parsed; return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "evo-volume",
                "eco",
                "name",
                "pgn",
                "uci",
                "epd",
                "epdhash",
            ]
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