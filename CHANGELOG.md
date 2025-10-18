# Changelog

## 0.1.1 (2025-10-18)

- Files starting with a UTF-8 byte order mark (Excel, some Windows exporters) no longer report
  their first line as invalid JSON

## 0.1.0 (2025-10-09)

First release.

- Readers and writers for OpenAI, Anthropic and ShareGPT conversation formats, auto-detected
  per line; `convert` between any two
- 18 conversation rules: structure, tool call/result pairing, arguments against JSON Schema,
  PII, template leftovers, canned disclaimers, token length
- Dataset rules: exact duplicates and MinHash/LSH near-duplicates, across files
- `check` with text, JSON and GitHub annotation output and `--statistics`; `dedupe`; `stats`
- Settings in `[tool.convlint]`; pre-commit hook; GitHub Action
