from __future__ import annotations

from collections import Counter
from pathlib import Path

import click

from . import __version__
from .check import check_file
from .formats import READERS, read_auto
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


@main.command("rules")
def list_rules() -> None:
    """List every rule with its default severity."""
    width = max(map(len, RULES))
    for r in RULES.values():
        click.echo(f"{r.name:<{width}}  {r.severity:<7}  {r.summary}")
