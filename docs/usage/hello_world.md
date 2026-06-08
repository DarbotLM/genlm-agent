# Hello world

Run the CLI against a simple observation:

```bash
python -m glmagent.run.run verify-observation --text "repo tests passed at 2026-03-09T12:00:00Z" --scope repo
```

Example output:

```json
[
  {
    "source": "cli",
    "timestamp": "2026-03-09T12:00:00+00:00",
    "data_timestamp": "2026-03-09T12:00:00+00:00",
    "level": "functional",
    "content": "repo tests passed at 2026-03-09T12:00:00Z",
    "scope_item": "repo"
  }
]
```
