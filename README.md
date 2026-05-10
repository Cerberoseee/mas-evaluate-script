## mas-evaluate-pipeline

Evaluator and experiment runner for a communication-focused SWE-bench Lite study.

### Study shape

- `mas_centralize`
- `mas_decentralized`

The evaluator keeps prompt fields, timeout, token budget, workspace prep, and grading flow aligned across all arms.

### CLI

```bash
uv run mas-evaluate prepare-suite --manifest-path manifests/swebench_lite_25.json --dataset-jsonl path/to/swebench_lite.jsonl --count 25
uv run mas-evaluate run-benchmark --config study.example.toml --manifest-path manifests/swebench_lite_25.json
uv run mas-evaluate grade-runs --config study.example.toml --manifest-path manifests/swebench_lite_25.json --run-id example-study
uv run mas-evaluate report --config study.example.toml --run-id example-study
```

### Config

Use [study.example.toml](/Users/cerberose/Work/home/cerberose/mas-evaluate-pipeline/study.example.toml) as the starting point.

Important knobs:

- `env_file` for secrets like `OPENAI_API_KEY`
- shared `base_model`
- `repeats`
- enabled `arms`
- harness repo mirrors / grading command
- per-arm command adapters for both MAS repos

`mini_swe_agent` is still supported by the pipeline, but the default thesis configuration is MAS-only.

### Artifacts

- `manifests/`: frozen task subsets
- `runs/<run_id>/<arm>/<task_id>/<repeat>/`: prompt, task context, stdout/stderr, patch, telemetry, grade result
- `reports/<run_id>/`: Markdown, JSON, and CSV summaries

### Local verification

The test suite uses fake adapter and grader scripts so the evaluator can be exercised without the real SWE-bench harness or model backends:

```bash
pytest -q
```

### Environment file

The study config can point at an env file:

```toml
env_file = ".env"
```

Example:

```env
OPENAI_API_KEY=sk-...
```

The CLI loads that file before any benchmark arm runs. A config-level
`openai_api_key` still works too, but `.env` is the cleaner default.
