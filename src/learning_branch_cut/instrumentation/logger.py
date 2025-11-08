from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

from rich.console import Console
from rich.table import Table


@dataclass
class NodeLogRecord:
    node_id: int
    depth: int
    dual_bound: float
    primal_bound: float
    gap: float
    fractional_variables: int
    timestamp: float
    outcome: Optional[str] = None
    estimate: Optional[float] = None
    solving_time: Optional[float] = None
    lp_iterations: Optional[int] = None
    nodes_processed: Optional[int] = None
    nodes_in_queue: Optional[int] = None
    cuts_applied: Optional[int] = None
    node_type: Optional[str] = None
    strong_branch_score: Optional[float] = None
    strong_branch_label: Optional[int] = None


@dataclass
class CutLogRecord:
    cut_id: int
    separator: str
    violation: float
    efficacy: float
    sparsity: float
    application_time: float
    dual_improvement: Optional[float] = None
    node_id: Optional[int] = None
    solving_time: Optional[float] = None
    lp_iterations: Optional[int] = None
    cuts_applied_total: Optional[int] = None
    lp_rows: Optional[int] = None
    lp_cols: Optional[int] = None
    round_index: Optional[int] = None


class TelemetryLogger:
    """Collects and persists telemetry emitted by the solver layer."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.node_records: List[NodeLogRecord] = []
        self.cut_records: List[CutLogRecord] = []
        self.console = Console()

    def record_node(self, record: NodeLogRecord) -> None:
        self.node_records.append(record)

    def record_cut(self, record: CutLogRecord) -> None:
        self.cut_records.append(record)

    def flush(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(self.output_dir / "node_logs.jsonl", self.node_records)
        _write_jsonl(self.output_dir / "cut_logs.jsonl", self.cut_records)

    def summary(self) -> None:
        node_table = Table(title="Node Records", show_edge=False, header_style="bold white")
        for column in [
            "node_id",
            "depth",
            "gap",
            "sb",
            "frac_vars",
            "lp_iters",
            "queue",
            "outcome",
        ]:
            node_table.add_column(column)
        for record in self.node_records[-10:]:
            node_table.add_row(
                str(record.node_id),
                str(record.depth),
                f"{record.gap:.4f}",
                "-" if record.strong_branch_label is None else str(record.strong_branch_label),
                str(record.fractional_variables),
                "-" if record.lp_iterations is None else str(record.lp_iterations),
                "-" if record.nodes_in_queue is None else str(record.nodes_in_queue),
                record.outcome or "-",
            )

        cut_table = Table(title="Cut Records", show_edge=False, header_style="bold white")
        for column in [
            "cut_id",
            "separator",
            "node_id",
            "violation",
            "efficacy",
            "lp_iters",
            "dual_improvement",
        ]:
            cut_table.add_column(column)
        for record in self.cut_records[-10:]:
            cut_table.add_row(
                str(record.cut_id),
                record.separator,
                "-" if record.node_id is None else str(record.node_id),
                f"{record.violation:.3f}",
                f"{record.efficacy:.3f}",
                "-" if record.lp_iterations is None else str(record.lp_iterations),
                "-"
                if record.dual_improvement is None
                else f"{record.dual_improvement:.4f}",
            )

        self.console.print(node_table)
        self.console.print(cut_table)


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    import json

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)))
            handle.write("\n")
