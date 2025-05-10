from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click

from . import __version__
from .check import check_file
from .formats import READERS, WRITERS, FormatError, read_auto
from .io import read_jsonl
from .rules import RULES

_existing_file = click.Path(exists=True, dir_okay=False, path_type=Path)
_format_option = click.option(
    "--format",
    "fmt",
    type=click.Choice(["auto", *READERS]),
    default="auto",
    show_default=True,
    help="Input format; auto-detected per line by default.",
)


def _rule_list(ctx: click.Context, param: click.Parameter, value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in RULES]
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
@_format_option
def check(
    files: tuple[Path, ...], select: list[str] | None, ignore: list[str] | None, strict: bool, fmt: str
) -> None:
    """Lint JSONL datasets. Exits 1 if any errors are found."""
    reader = read_auto if fmt == "auto" else READERS[fmt]
    severities: Counter[str] = Counter()
    for path in files:
        for f in check_file(path, reader, select, ignore or ()):
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
def convert(source: Path, target: str, output, fmt: str) -> None:
    """Convert a JSONL dataset between OpenAI, Anthropic and ShareGPT formats.

    Lines that can't be read or converted are reported on stderr and left out; the exit code
    is 1 if there were any.
    """
    reader = read_auto if fmt == "auto" else READERS[fmt]
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


@main.command("rules")
def list_rules() -> None:
    """List every rule with its default severity."""
    width = max(map(len, RULES))
    for r in RULES.values():
        click.echo(f"{r.name:<{width}}  {r.severity:<7}  {r.summary}")
