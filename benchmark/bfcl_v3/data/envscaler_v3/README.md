# EnvScaler BFCL-v3 multi-turn input snapshot

Source: https://github.com/RUC-NLPIR/EnvScaler/tree/7ab81e557c5992d62d4533e81bfec3d2ca0ae24b/evaluation/bfcl_eval/data

These four JSONL files contain 200 tasks per category (800 total).
Only these question files differ from the project's pinned official BFCL export.
Function documentation and possible answers were checked byte-for-byte against
the EnvScaler snapshot and remain in `benchmark/bfcl_v3/official/bfcl_eval/data`.

Select this snapshot with the existing entry point's
`--bfcl-data-dir benchmark/bfcl_v3/data/envscaler_v3` option. With the option
omitted, the existing official dataset remains the default.

The official Qwen handler, API calls, environment execution, and scoring are
unchanged. EnvScaler's source license is preserved in `LICENSE`.
