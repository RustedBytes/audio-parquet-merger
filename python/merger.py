import argparse
import polars as pl
from pathlib import Path
from typing import NoReturn


def merge_parquets(input1_path: str, input2_path: str, output_path: str) -> NoReturn:
    """
    Merge two Parquet files containing audio data, joining on the 'path' field
    within the 'audio' struct column.
    The 'audio' struct from the first file is renamed to 'audio_1', and the
    'audio' struct from the second file is added as 'audio_2'. All other columns
    from the first file are preserved.

    Args:
        input1_path: Path to the first Parquet file.
        input2_path: Path to the second Parquet file.
        output_path: Path to the output Parquet file.

    Raises:
        FileNotFoundError: If an input file does not exist.
        ValueError: If required 'audio' column is missing or lacks 'path' field in either file.
    """
    input1 = Path(input1_path)
    input2 = Path(input2_path)
    output = Path(output_path)

    if not input1.exists():
        raise FileNotFoundError(f"Input file not found: {input1_path}")
    if not input2.exists():
        raise FileNotFoundError(f"Input file not found: {input2_path}")

    df1 = pl.read_parquet(input1_path)
    audio_dtype1 = df1.schema.get("audio")
    if audio_dtype1 is None:
        raise ValueError("Column 'audio' not found in first file")
    if not isinstance(audio_dtype1, pl.Struct):
        raise ValueError("Column 'audio' must be a struct dtype in first file")
    if "path" not in [f.name for f in audio_dtype1.fields]:
        raise ValueError("Field 'path' not found in 'audio' struct of first file")

    df2 = pl.read_parquet(input2_path)
    audio_dtype2 = df2.schema.get("audio")
    if audio_dtype2 is None:
        raise ValueError("Column 'audio' not found in second file")
    if not isinstance(audio_dtype2, pl.Struct):
        raise ValueError("Column 'audio' must be a struct dtype in second file")
    if "path" not in [f.name for f in audio_dtype2.fields]:
        raise ValueError("Field 'path' not found in 'audio' struct of second file")

    # Select only aliased audio from second file (full struct as 'audio_2')
    audio2_df = df2.select(pl.col("audio").alias("audio_2"))

    # Inner join on audio.path from both, then rename audio from first to audio_1
    merged_df = df1.join(
        audio2_df,
        left_on=pl.col("audio").struct.field("path"),
        right_on=pl.col("audio_2").struct.field("path"),
        how="inner",
    ).rename({"audio": "audio_1"})

    # Reorder columns to have audio_1, audio_2, then the rest
    other_cols = [col for col in df1.columns if col != "audio"]
    # The join adds a 'path_right' column which we should drop
    final_cols = ["audio_1", "audio_2"] + other_cols
    merged_df = merged_df.select(final_cols)

    merged_df.write_parquet(output)
    print(f"Merged data written to {output_path} ({len(merged_df)} rows)")


def main() -> NoReturn:
    """CLI entrypoint for merging Parquet audio files."""
    parser = argparse.ArgumentParser(
        description="Merge two Parquet files on 'audio.path' field, creating audio_1 and audio_2 columns."
    )
    parser.add_argument(
        "input1",
        type=str,
        help="Path to the first Parquet file (becomes base with audio_1)",
    )
    parser.add_argument(
        "input2", type=str, help="Path to the second Parquet file (provides audio_2)"
    )
    parser.add_argument("output", type=str, help="Path to the output Parquet file")
    args = parser.parse_args()

    try:
        merge_parquets(args.input1, args.input2, args.output)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import sys

    main()
