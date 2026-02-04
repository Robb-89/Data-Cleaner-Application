# Data Cleaner

A robust Python script for cleaning, deduplicating, and validating contact data from large CSV or Excel files. Handles messy input, missing columns, and malformed rows. Provides chunked processing, progress reporting, error summaries, and detailed missing/invalid reports.

## Features
- Cleans and deduplicates contact data (names, emails, phones, addresses)
- Handles large files with chunked processing
- Graceful interrupt handling (Ctrl+C)
- Real-time progress reporting
- Outputs cleaned data, error log, summary, and missing/invalid report
- Supports CSV and Excel (.xlsx) input
- Flexible column mapping and synonym detection
- Automatically fixes row-length issues in CSVs
- Always outputs blank columns for missing essentials (e.g., name_first)

## Usage

### Basic Command
```sh
python data_cleaner.py csv --path <input.csv> --out <output_prefix>
python data_cleaner.py excel --path <input.xlsx> --sheet <SheetName> --out <output_prefix>
```

### Options
- `--path <file>`: Path to input CSV or Excel file (required)
- `--out <prefix>`: Output file prefix (default: cleaned_output)
- `--dedupe <mode>`: Deduplication mode (`email`, `name_phone`, `smart`, `none`; default: smart)
- `--chunksize <N>`: Process CSV in chunks of N rows (recommended for large files)
- `--keep_original`: Include all original columns in output (default: False)
- `--drop_cols <col1,col2,...>`: Exclude columns from cleaned output
- `--no_xlsx`: Skip writing Excel output
- `--sheet <SheetName>`: Excel sheet name (required for Excel mode)

### Example
```sh
python data_cleaner.py csv --path messy.csv --out sample_test --chunksize 10000
python data_cleaner.py excel --path contacts.xlsx --sheet Sheet1 --out cleaned_contacts
```

## Output Files
- `<prefix>.csv`: Cleaned contacts
- `<prefix>.xlsx`: Cleaned contacts (Excel; only in full mode)
- `<prefix>_errors.csv`: Row-by-row error log
- `<prefix>_summary.txt`: Summary of counts and error types
- `<prefix>_missing_report.csv`: Detailed report of contacts missing/invalid required fields

## Requirements
- Python 3.7+
- pandas
- numpy
- openpyxl (for Excel input/output)

Install dependencies:
```sh
pip install pandas numpy openpyxl
```

## Notes
- For very large files, use `--chunksize` to avoid memory issues.
- If columns like "First Name" are missing, blank columns will be added automatically.
- Interrupting the script (Ctrl+C) will write a partial summary.
- All outputs are written to the current directory unless otherwise specified.

## Troubleshooting
- If you see a PermissionError, close the output file in Excel and re-run.
- If you get a pandas error about row length, the script will automatically fix and process all rows.

## License
MIT
