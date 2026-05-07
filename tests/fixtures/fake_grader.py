from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    predictions_path = Path(sys.argv[1])
    grade_path = Path(sys.argv[2])
    lines = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    resolved = bool(lines and lines[0].get("model_patch"))
    grade_path.write_text(
        json.dumps({"resolved": resolved, "status": "graded", "predictions": len(lines)}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
