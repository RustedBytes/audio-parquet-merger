use anyhow::{Context, Result, anyhow};
use clap::Parser;
use polars::prelude::*;
use polars_parquet::parquet::compression::Compression;
use polars_parquet::read::read_metadata;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Merge two Parquet files on 'audio.path' field, creating audio_1 and audio_2 columns.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the first Parquet file (becomes base with audio_1)
    #[arg(value_name = "INPUT_1")]
    input1: PathBuf,

    /// Path to the second Parquet file (provides audio_2)
    #[arg(value_name = "INPUT_2")]
    input2: PathBuf,

    /// Path to the output Parquet file
    #[arg(value_name = "OUTPUT")]
    output: PathBuf,
}

/// Validates that the DataFrame schema contains an 'audio' struct with a 'path' field.
fn validate_audio_schema(df: &DataFrame, file_path: &Path) -> Result<()> {
    let schema = df.schema();
    let audio_field = schema
        .get_field("audio")
        .ok_or_else(|| anyhow!("Column 'audio' not found in file: {}", file_path.display()))?;

    if let DataType::Struct(fields) = audio_field.dtype() {
        if !fields.iter().any(|f| f.name() == "path") {
            return Err(anyhow!(
                "Field 'path' not found in 'audio' struct of file: {}",
                file_path.display()
            ));
        }
    } else {
        return Err(anyhow!(
            "Column 'audio' must be a struct dtype in file: {}",
            file_path.display()
        ));
    }
    Ok(())
}

/// Reads metadata from a Parquet file and returns the compression setting for the first column.
fn get_compression_from_parquet(path: &Path) -> Result<Compression> {
    let mut file = File::open(path)?;
    let metadata = read_metadata(&mut file)
        .with_context(|| format!("Failed to read metadata from {}", path.display()))?;

    let compression: Compression = metadata
        .row_groups
        .first()
        .and_then(|rg| rg.parquet_columns().first())
        .map(|col| col.compression())
        .ok_or_else(|| anyhow!("Could not determine compression from {}", path.display()))?;

    Ok(compression)
}

/// Merge two Parquet files containing audio data, joining on the 'path' field
/// within the 'audio' struct column.
fn merge_parquets(input1_path: &Path, input2_path: &Path, output_path: &Path) -> Result<()> {
    let file = File::open(input1_path)?;
    let df1 = ParquetReader::new(file).finish()?;
    validate_audio_schema(&df1, input1_path)?;

    // Read and validate second dataframe
    let file = File::open(input2_path)?;
    let df2 = ParquetReader::new(file).finish()?;
    validate_audio_schema(&df2, input2_path)?;

    // Prepare df2 for joining: select and alias 'audio' to 'audio_2'
    // and also select the join key.
    let df2_for_join = df2
        .lazy()
        .select([
            col("audio").alias("audio_2"),
            col("audio").struct_().field_by_name("path"),
        ])
        .collect()?;

    // Inner join on audio.path from both
    let merged_df = df1
        .lazy()
        .join(
            df2_for_join.lazy(),
            [col("audio").struct_().field_by_name("path")],
            [col("path")],
            JoinArgs::new(JoinType::Inner),
        )
        .collect()?;

    // Rename 'audio' to 'audio_1' and select columns in the final order
    let mut final_df = merged_df.clone();
    final_df.rename("audio", "audio_1".into())?;

    let final_cols = vec!["audio_1", "audio_2", "duration", "transcription"];
    let mut final_df = final_df.select(final_cols)?;

    // Write the merged dataframe to a new Parquet file
    let mut output_file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {}", output_path.display()))?;

    // Get compression algorithm from the first input file to apply to the output.
    let cmp = get_compression_from_parquet(input1_path)?;

    let pq_compression = match cmp {
        Compression::Uncompressed => ParquetCompression::Uncompressed,
        Compression::Snappy => ParquetCompression::Snappy,
        Compression::Gzip => ParquetCompression::Gzip(None),
        Compression::Lzo => ParquetCompression::Lzo,
        Compression::Brotli => ParquetCompression::Brotli(None),
        Compression::Lz4 => ParquetCompression::Lzo,
        Compression::Zstd => ParquetCompression::Zstd(None),
        Compression::Lz4Raw => ParquetCompression::Lz4Raw,
    };

    ParquetWriter::new(&mut output_file)
        .with_compression(pq_compression)
        .with_row_group_size(Some(256))
        .finish(&mut final_df)?;

    println!(
        "Merged data written to {} ({} rows)",
        output_path.display(),
        final_df.height()
    );

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.input1.exists() {
        return Err(anyhow!("Input file not found: {}", args.input1.display()));
    }
    if !args.input2.exists() {
        return Err(anyhow!("Input file not found: {}", args.input2.display()));
    }

    merge_parquets(&args.input1, &args.input2, &args.output)
        .with_context(|| "Failed to merge Parquet files")
}
