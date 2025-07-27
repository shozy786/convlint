from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click

from . import __version__
from . import config
from .check import check_files
from .dataset import DATASET_RULES, ExactDuplicates, NearDuplicates
from .formats import READERS, WRITERS, FormatError, read_auto
from .io import read_jsonl
from .rules import RULES, Options

_existing_file = click.Path(exists=True, dir_okay=False, path_type=Path)
_format_option = click.option(
    "--format",
    "fmt",
    type=click.Choice(["auto", *READERS]),
    help="Input format; auto-detected per line unless configured.",
)


def _settings(path: Path | None) -> config.Settings:
    try:
        return config.load(path or config.find())
    except config.ConfigError as e:
        raise click.ClickException(str(e)) from None


def _reader(fmt: str | None, settings: config.Settings):
    fmt = fmt or settings.format
    return read_auto if fmt == "auto" else READERS[fmt]


def _rule_list(ctx: click.Context, param: click.Parameter, value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in RULES and n not in DATASET_RULES]
    if unknown:
        raise click.BadParameter(f"unknown rules {unknown}; see `convlint rules`")
    return names


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Lint and convert tool-calling conversation datasets."""


@main.command()
@click.argument("files", nargs=-1, required=True, type=_existing_file)
@click.option("--select", callback=_rule_list, help="Only run these rules (comma separated).")
@click.option("--ignore", callback=_rule_list, help="Skip these rules (comma separated).")
@click.option("--strict", is_flag=True, help="Exit non-zero on warnings too.")
@click.option("--config", "config_file", type=_existing_file, help="Settings file (default: nearest pyproject.toml).")
@_format_option
def check(
    files: tuple[Path, ...],
    select: list[str] | None,
    ignore: list[str] | None,
    strict: bool,
    config_file: Path | None,
    fmt: str | None,
) -> None:
    """Lint JSONL datasets. Exits 1 if any errors are found.

    Settings come from [tool.convlint] in the nearest pyproject.toml (or a convlint.toml);
    --select replaces the configured selection, --ignore adds to it.
    """
    settings = _settings(config_file)
    select = select if select is not None else settings.select
    ignore = [*settings.ignore, *(ignore or ())]
    severities: Counter[str] = Counter()
    for f in check_files(files, _reader(fmt, settings), select, ignore, settings.options):
        severities[f.problem.severity] += 1
        turn = f" [turn {f.problem.turn}]" if f.problem.turn is not None else ""
        click.echo(f"{f.path}:{f.line}: {f.problem.severity} {f.problem.rule}{turn} {f.problem.message}")
    errors, warnings = severities["error"], severities["warning"]
    click.echo(f"{errors} error{'s' * (errors != 1)}, {warnings} warning{'s' * (warnings != 1)}", err=True)
    if errors or (strict and warnings):
        raise SystemExit(1)


@main.command()
@click.argument("source", type=_existing_file)
@click.option("--to", "target", type=click.Choice(sorted(WRITERS)), required=True, help="Output format.")
@click.option("-o", "--output", type=click.File("w", encoding="utf-8"), default="-", help="Output file (default stdout).")
@_format_option
def convert(source: Path, target: str, output, fmt: str | None) -> None:
    """Convert a JSONL dataset between OpenAI, Anthropic and ShareGPT formats.

    Lines that can't be read or converted are reported on stderr and left out; the exit code
    is 1 if there were any.
    """
    reader = _reader(fmt, _settings(None))
    writer = WRITERS[target]
    written = failed = 0
    for line in read_jsonl(source):
        try:
            if line.error is not None:
                raise FormatError(line.error)
            output.write(json.dumps(writer(reader(line.record)), ensure_ascii=False) + "\n")
            written += 1
        except FormatError as e:
            click.echo(f"{source}:{line.number}: skipped, {e}", err=True)
            failed += 1
    click.echo(f"converted {written} conversations to {target}" + (f", skipped {failed}" if failed else ""), err=True)
    if failed:
        raise SystemExit(1)


@main.command()
@click.argument("source", type=_existing_file)
@click.option("-o", "--output", type=click.File("w", encoding="utf-8"), required=True, help="Where to write kept lines.")
@click.option("--threshold", type=click.FloatRange(0.5, 1.0), default=0.85, show_default=True, help="Near-duplicate similarity.")
@click.option("--exact-only", is_flag=True, help="Only drop exact duplicates.")
@_format_option
def dedupe(source: Path, output, threshold: float, exact_only: bool, fmt: str | None) -> None:
    """Write SOURCE without duplicate and near-duplicate conversations, keeping the first of each.

    Kept lines are copied verbatim. Lines that can't be parsed are kept too (fixing them is
    `check`'s job) and counted on stderr.
    """
    reader = _reader(fmt, _settings(None))
    options = Options(near_duplicate_threshold=threshold)
    checks = [ExactDuplicates(options)] + ([] if exact_only else [NearDuplicates(options)])
    kept = dropped = unreadable = 0
    for line in read_jsonl(source):
        try:
            if line.record is None:
                raise FormatError(line.error or "")
            conversation = reader(line.record)
        except FormatError:
            unreadable += 1
        else:
            problems = [p for check in checks for p in check.observe((str(source), line.number), conversation)]
            if problems:
                dropped += 1
                continue
        output.write(line.text if line.text.endswith("\n") else line.text + "\n")
        kept += 1
    note = f", {unreadable} unreadable lines kept as-is" if unreadable else ""
    click.echo(f"kept {kept}, dropped {dropped} duplicates{note}", err=True)


@main.command("rules")
def list_rules() -> None:
    """List every rule with its default severity."""
    every = [*RULES.values(), *DATASET_RULES.values()]
    width = max(len(r.name) for r in every)
    for r in every:
        click.echo(f"{r.name:<{width}}  {r.severity:<7}  {r.summary}")
