"""Live viva demo: one document end-to-end, every model vs gold side-by-side.

Modes:
  --replay (default)  read results/runs.jsonl — instant, deterministic, cannot
                      fail on a flaky network or slow Ollama load.
  --live              actually call models on the chosen document.
  --live --local-only call only the smallest local model (skip Azure).

Fallback order for the viva: --replay  ->  --live --local-only  ->  --live
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.data.gt import align_gold, load_test_records, schema_for_record  # noqa: E402
from src.data.preprocess import from_dataset_record  # noqa: E402
from src.eval.metrics import score_document  # noqa: E402
from src.eval.parse import parse_prediction  # noqa: E402
from src.runner.results_store import load_rows  # noqa: E402

console = Console()

_CAT_STYLE = {
    "tp": ("[green]✓ match[/green]"),
    "wrong": ("[red]✗ wrong[/red]"),
    "missing": ("[yellow]– missing[/yellow]"),
    "hallucinated": ("[magenta]+ hallucinated[/magenta]"),
    "tn": ("[dim]· both empty[/dim]"),
}


def _pick_record(records: list[dict], doc_id: str | None) -> dict:
    if doc_id:
        for r in records:
            if r["doc_id"] == doc_id:
                return r
        console.print(f"[red]doc_id {doc_id!r} not found.[/red]")
        sys.exit(1)
    # Default: a receipt (clean, readable for a live audience), else first doc.
    for r in records:
        if r.get("schema_name") == "sroie":
            return r
    return records[0]


def _show_document(record: dict, simple_json: dict) -> None:
    lines = "\n".join(simple_json["lines"]) or "(none)"
    kv = "\n".join(f"{k}: {v}" for k, v in simple_json["key_values"].items()) or "(none)"
    body = f"[bold]LINES[/bold]\n{lines}\n\n[bold]KEY_VALUES[/bold]\n{kv}"
    console.print(Panel(body, title=f"[cyan]Document {record['doc_id']}[/cyan] "
                                    f"(dataset: {record['dataset']})", expand=False))


def _model_table(model_id: str, gold: dict, pred: dict | None, schema: dict,
                 latency_s, cost_usd, f1, verbose: bool = False) -> Table:
    score = score_document(gold, pred, schema)
    head = f"[bold]{model_id}[/bold]  F1={f1:.2f}  lat={latency_s}s  ${cost_usd:.5f}"
    if verbose:
        head += (f"\n[dim]P={score['precision']:.2f}  R={score['recall']:.2f}  "
                 f"exact={score['exact_match_fields']}/{score['n_fields']} fields"
                 f"{'  (all correct)' if score['exact_match_doc'] else ''}[/dim]")
    table = Table(title=head, title_justify="left", expand=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Gold")
    table.add_column("Predicted")
    table.add_column("Status")
    for field in [f["name"] for f in schema["fields"]]:
        g = "" if gold.get(field) is None else str(gold.get(field))
        p = "" if (pred is None or pred.get(field) is None) else str(pred.get(field))
        cat = score["per_field"][field]
        table.add_row(field, g, p, _CAT_STYLE.get(cat, cat))
    return table


def _detail_panel(model_id: str, *, shot_mode: str, input_variant: str,
                  prompt_tokens=None, completion_tokens=None, peak_mem_mb=None,
                  raw_text: str | None = None) -> Panel:
    """Under-the-hood detail for --verbose: condition, tokens, memory, raw output."""
    parts = [
        f"[bold]condition[/bold]  shot={shot_mode}  input={input_variant}",
        f"[bold]tokens[/bold]     prompt={prompt_tokens}  completion={completion_tokens}",
        f"[bold]peak mem[/bold]   {peak_mem_mb} MB",
    ]
    if raw_text is not None:
        shown = raw_text if len(raw_text) <= 800 else raw_text[:800] + " …[truncated]"
        parts.append(f"[bold]raw output[/bold]\n[dim]{shown or '(empty)'}[/dim]")
    return Panel("\n".join(parts), title=f"[cyan]{model_id} — detail[/cyan]",
                 style="dim", expand=False)


def replay(cfg: dict, doc_id: str | None, models: list[str] | None,
           verbose: bool = False) -> None:
    rows = load_rows(resolve_path(cfg, cfg["paths"]["runs_jsonl"]))
    if not rows:
        console.print("[red]No results in runs.jsonl.[/red] Seed them first, e.g.:\n"
                      "  python scripts/run_eval.py --pilot --mock")
        sys.exit(1)

    records = load_test_records(cfg)
    record = _pick_record(records, doc_id)
    schema = schema_for_record(cfg, record)
    gold = align_gold(record.get("gold", {}), schema)
    simple_json = from_dataset_record(record)

    console.print(Panel("[bold]REPLAY[/bold] — from saved results (results/runs.jsonl)",
                        style="green", expand=False))
    _show_document(record, simple_json)

    doc_rows = [r for r in rows if r["doc_id"] == record["doc_id"]]
    seen = {}
    for r in doc_rows:
        if models and r["model_id"] not in models:
            continue
        seen.setdefault(r["model_id"], r)  # first condition per model
    if not seen:
        console.print("[yellow]No saved rows for this document/model selection.[/yellow]")
        return
    for model_id, r in seen.items():
        console.print(_model_table(model_id, gold, r.get("predicted_json"), schema,
                                   r.get("latency_s"), r.get("cost_usd") or 0.0,
                                   r.get("f1") or 0.0, verbose=verbose))
        if verbose:
            console.print(_detail_panel(
                model_id, shot_mode=r.get("shot_mode", "?"),
                input_variant=r.get("input_variant", "?"),
                prompt_tokens=r.get("prompt_tokens"),
                completion_tokens=r.get("completion_tokens"),
                peak_mem_mb=r.get("peak_mem_mb"),
                raw_text=None))  # raw text not stored in runs.jsonl; show predicted JSON via table


def _preflight_live(cfg, runners) -> list:
    ok = []
    for runner in runners:
        try:
            runner.ensure_available()
            ok.append(runner)
            console.print(f"[green]✓[/green] {runner.model_id} ready ({runner.kind})")
        except Exception as exc:
            console.print(f"[yellow]✗ {runner.model_id} unavailable:[/yellow] {exc}")
    return ok


def live(cfg: dict, doc_id: str | None, models: list[str] | None, local_only: bool,
         verbose: bool = False, show_prompt: bool = False) -> None:
    from src.eval.efficiency import efficiency_metrics
    from src.models.registry import build_runners, model_meta

    records = load_test_records(cfg)
    record = _pick_record(records, doc_id)
    schema = schema_for_record(cfg, record)
    gold = align_gold(record.get("gold", {}), schema)
    simple_json = from_dataset_record(record)

    if not models:
        # smallest local + frontier API by default
        local_ids = [m["id"] for m in cfg["models"] if m["type"] == "local"]
        api_ids = [m["id"] for m in cfg["models"] if m["type"] == "api"]
        models = ([local_ids[0]] if local_ids else [])
        if not local_only and api_ids:
            models.append(api_ids[0])

    runners = build_runners(cfg, only=models)
    meta = model_meta(cfg)

    console.print(Panel("[bold]LIVE[/bold] — calling models now", style="red", expand=False))
    runners = _preflight_live(cfg, runners)
    if not runners:
        console.print("[red]No models available. Falling back to --replay is recommended.[/red]")
        sys.exit(1)
    _show_document(record, simple_json)

    from src.prompts.builder import build_prompt
    shot_mode, input_variant = "zero_shot", "lines_plus_kv"
    prompt = build_prompt(schema, simple_json, shot_mode, input_variant,
                          max_lines=cfg["conditions"].get("max_input_lines"))
    if show_prompt:
        console.print(Panel(prompt, title="[cyan]Prompt sent to every model[/cyan]",
                            style="dim", expand=False))
    for runner in runners:
        with console.status(f"Running {runner.model_id}…"):
            result = runner.run(prompt)
        pred, _ = parse_prediction(result.text, schema)
        eff = efficiency_metrics(result, meta[runner.model_id], runner.kind,
                                 cfg.get("prices_per_1m_tokens", True))
        f1 = score_document(gold, pred, schema)["f1"]
        if result.error:
            console.print(f"[red]{runner.model_id} error:[/red] {result.error}")
            continue
        console.print(_model_table(runner.model_id, gold, pred, schema,
                                   eff["latency_s"], eff["cost_usd"], f1, verbose=verbose))
        if verbose:
            console.print(_detail_panel(
                runner.model_id, shot_mode=shot_mode, input_variant=input_variant,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                peak_mem_mb=result.peak_mem_mb, raw_text=result.text))


def main() -> None:
    ap = argparse.ArgumentParser(description="SLM-vs-LLM KIE viva demo.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--replay", action="store_true", help="From saved results (default).")
    mode.add_argument("--live", action="store_true", help="Call models live.")
    ap.add_argument("--doc", default=None, help="Specific doc_id to show.")
    ap.add_argument("--models", nargs="*", default=None, help="Restrict to these model ids.")
    ap.add_argument("--local-only", action="store_true", help="(live) skip the Azure model.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show P/R/exact-match, condition, tokens, memory, and raw output.")
    ap.add_argument("--show-prompt", action="store_true",
                    help="(live) print the full prompt sent to the models.")
    args = ap.parse_args()

    cfg = load_config()
    if args.live:
        live(cfg, args.doc, args.models, args.local_only, args.verbose, args.show_prompt)
    else:
        replay(cfg, args.doc, args.models, args.verbose)


if __name__ == "__main__":
    main()
