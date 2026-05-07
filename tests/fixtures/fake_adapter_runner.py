from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    run_dir = Path(os.environ["MAS_EVAL_RUN_DIR"])
    patch_path = Path(os.environ["MAS_EVAL_PATCH_PATH"])
    telemetry_path = Path(os.environ["MAS_EVAL_TELEMETRY_PATH"])
    result_path = Path(os.environ["MAS_EVAL_RESULT_PATH"])
    behavior = os.environ.get("FAKE_RUNNER_BEHAVIOR", "success")
    if behavior == "fail":
        raise SystemExit(7)
    if behavior != "nopatch":
        patch_path.write_text("diff --git a/foo.py b/foo.py\n", encoding="utf-8")
    telemetry_path.write_text(
        json.dumps(
            {
                "total_tokens": 123,
                "prompt_tokens": 100,
                "completion_tokens": 23,
                "total_steps": 4,
                "tool_calls": 3,
                "tool_failures": 1,
                "retries": 1,
                "messages": 5,
                "handoffs": 2,
            }
        ),
        encoding="utf-8",
    )
    result_path.write_text(json.dumps({"status": "ok", "run_dir": str(run_dir)}), encoding="utf-8")


if __name__ == "__main__":
    main()
