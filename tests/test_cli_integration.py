import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_cli_integration(tmp_path):
    """
    Full integration test:
    - create a temporary returns.csv
    - run run_mecd.py through subprocess (as a user would)
    - verify output CSV is created
    - verify shape and content are valid
    """

    # 1. Build temp CSV with random data
    dates = pd.date_range("2020-01-01", periods=150)  # enough for EWMA warmup
    data = np.random.normal(0.0005, 0.01, size=(150, 4))
    df = pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])

    input_path = tmp_path / "returns.csv"
    output_path = tmp_path / "mecd_signal.csv"

    df.to_csv(input_path)

    # 2. Run the CLI script
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_mecd.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )

    # CLI should exit normally
    assert result.returncode == 0, f"CLI crashed: {result.stderr}"

    # 3. Output file must exist
    assert output_path.exists(), "CLI did not create output CSV"

    # 4. Output must be a valid DataFrame
    out = pd.read_csv(output_path, index_col=0, parse_dates=True)

    assert out.shape[0] == df.shape[0], "Output rows mismatch"
    assert out.shape[1] == df.shape[1], "Output columns mismatch"

    # 5. There must be at least SOME finite values (after warmup)
    assert np.isfinite(out.values).sum() > 0, "CLI produced all-NaN output"
