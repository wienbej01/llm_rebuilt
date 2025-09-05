"""
Script to extract a sample of data from a larger dataset.
"""
import csv
import argparse


def extract_sample_data(input_file: str, output_file: str, num_rows: int):
    """
    Extracts a sample of data from a larger dataset.

    Args:
        input_file: Path to the input CSV file.
        output_file: Path to the output CSV file.
        num_rows: Number of rows to extract.
    """
    print(f"Extracting {num_rows} rows from {input_file} to {output_file}...")

    with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Write header
        header = next(reader)
        writer.writerow(header)

        # Write sample rows
        for i, row in enumerate(reader):
            if i >= num_rows:
                break
            writer.writerow(row)

    print("Extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a sample of data from a larger dataset.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("output_file", help="Path to the output CSV file.")
    parser.add_argument("--num_rows", type=int, default=1000, help="Number of rows to extract.")
    args = parser.parse_args()

    extract_sample_data(args.input_file, args.output_file, args.num_rows)
