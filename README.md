# convlint

Lint and convert multi-turn, tool-calling conversation datasets for supervised fine-tuning.

Fine-tuning data written by people, or generated and then edited by people, breaks in ways that
don't crash the trainer. A tool call that never gets a result, arguments that don't match the
tool's schema, a `{{customer_email}}` nobody filled in, the same templated conversation
four hundred times: the model learns all of it. convlint finds these before a training run does.

```
$ convlint check examples/support.jsonl
examples/support.jsonl:2: warning near-duplicate ~87% similar to examples/support.jsonl:1
examples/support.jsonl:3: error arguments-schema-mismatch [turn 2] refund_order: reason: 'torn' is not one of ['damaged', 'late', 'wrong_item', 'other']
examples/support.jsonl:4: error tool-call-pairing [turn 2] tool call 'call_2' never gets a result
examples/support.jsonl:5: warning pii [turn 1] email address sana…pk
examples/support.jsonl:5: error template-leftover [turn 2] unfilled placeholder '{{customer_email}}'
examples/support.jsonl:6: warning assistant-boilerplate [turn 2] canned phrase 'As an AI language model'; the model will learn to say it
examples/support.jsonl:7: error unknown-tool [turn 1] calls 'track_parcel', which is not among the declared tools ['lookup_order', 'refund_order']
4 errors, 3 warnings
```

## Install

```bash
pip install convlint            # character-based token estimates
pip install 'convlint[tokens]'  # exact counts for OpenAI encodings via tiktoken
```

Python 3.10+.

## Formats

Each line of a JSONL file is read in whichever of these formats it's in; mixed files are fine.

| format | shape |
|---|---|
| `openai` | `{"messages": [...], "tools": [...]}` with `tool_calls` / `tool_call_id`; legacy `functions` too |
| `anthropic` | `{"system": ..., "messages": [...], "tools": [...]}` with `tool_use` / `tool_result` content blocks |
| `sharegpt` | `{"conversations": [{"from", "value"}], "tools": [...]}` with `function_call` / `observation` turns (LLaMA-Factory, Axolotl) |

Pass `--format` to stop auto-detection.

## Commands

```bash
convlint check data/*.jsonl                       # lint; exit 1 on errors (--strict: on warnings too)
convlint check data.jsonl --output-format github  # annotations on the PR diff
convlint convert data.jsonl --to anthropic -o out.jsonl
convlint dedupe data.jsonl -o clean.jsonl         # drop exact and near-duplicates, keep the first
convlint stats data.jsonl                         # lengths, tool usage, token percentiles
convlint rules                                    # every rule and its severity
```

Conversion assigns stable ids (`call_<turn>_<n>`) when going from ShareGPT, which has none,
merges consecutive tool results into one user turn for Anthropic, and keeps any other keys on
the record (`id`, `source`, ...).

## Rules

| rule | severity | |
|---|---|---|
| `unknown-role` | error | Every message has a role of system, user, assistant or tool |
| `system-not-first` | error | A system message can only be the first message |
| `empty-message` | error | Messages have content (assistant messages may have only tool calls) |
| `consecutive-turns` | warning | User and assistant turns alternate |
| `no-assistant-turn` | error | There is at least one assistant message to train on |
| `bad-last-turn` | error | The conversation ends with an assistant message |
| `first-turn-not-user` | warning | The first non-system message comes from the user |
| `tool-call-pairing` | error | Every tool call gets exactly one result before the conversation moves on |
| `duplicate-tool-call-id` | error | Tool call ids are unique within a conversation |
| `tool-calls-outside-assistant` | error | Only assistant messages make tool calls |
| `invalid-tool-arguments` | error | Tool call arguments are a JSON object |
| `unknown-tool` | error | Tool calls only use tools declared on the conversation |
| `invalid-tool-schema` | error | Declared tool parameters are a valid JSON Schema |
| `arguments-schema-mismatch` | error | Tool call arguments validate against the tool's schema |
| `pii` | warning | No email addresses, phone numbers, card numbers or API keys in the text |
| `template-leftover` | error | No unfilled placeholders like `{{name}}`, `[INSERT DATE]` or `TODO` |
| `assistant-boilerplate` | warning | Assistant turns don't use canned AI disclaimers |
| `too-long` | warning | The conversation fits in the training context window (`max_tokens`) |
| `duplicate` | warning | No conversation appears twice, across all files checked |
| `near-duplicate` | warning | No two conversations are `near_duplicate_threshold` similar |

Notes on a few of them:

- **`pii`** masks what it finds (`sana…pk`) so CI logs don't become the leak. Card numbers must
  pass the Luhn check; documentation domains like `example.com` are ignored.
- **`near-duplicate`** compares user and assistant text with MinHash over word 3-shingles and
  LSH banding, so it's a single pass with constant memory per conversation. System prompts are
  left out, since they're usually shared by the whole dataset. Bands are chosen so a pair right at
  the threshold is caught 95% of the time.
- **`too-long`** counts tool schemas and calls too, since they're rendered into the prompt.
  Your trainer's tokenizer will differ a little; use a margin.

## Configuration

`[tool.convlint]` in the nearest `pyproject.toml`, or a `convlint.toml`:

```toml
[tool.convlint]
ignore = ["first-turn-not-user"]
max_tokens = 8192
encoding = "o200k_base"
near_duplicate_threshold = 0.9
format = "auto"
```

`--select` on the command line replaces the configured selection; `--ignore` adds to it.

## In CI

GitHub Actions, with findings as annotations on the PR diff:

```yaml
- uses: actions/checkout@v4
- uses: shozy786/convlint@v0.1.1
  with:
    paths: data/
    args: --strict
```

pre-commit:

```yaml
- repo: https://github.com/shozy786/convlint
  rev: v0.1.1
  hooks:
    - id: convlint
```

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest && ruff check && ruff format --check
```

## License

MIT
