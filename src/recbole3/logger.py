from __future__ import annotations

import dataclasses
import os
import time as time_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _is_main_process() -> bool:
    """Return True on global rank 0 or when not running under DDP."""
    rank = os.environ.get("RANK", "")
    if rank == "":
        return True
    try:
        return int(rank) in (-1, 0)
    except ValueError:
        return True


# Root for the browsable, model-organized log tree: log/<model>/<dataset>.log
CONSOLIDATED_LOG_ROOT = Path(os.environ.get("RECBOLE3_LOG_ROOT", "log"))

# Common Amazon-2014 category short tags used in experiment naming.
_CATEGORY_TAG_ALIASES = {
    "Beauty": "beauty",
    "Sports_and_Outdoors": "sports",
    "Toys_and_Games": "toys",
    "Cell_Phones_and_Accessories": "phones",
    "Clothing_Shoes_and_Jewelry": "clothing",
    "Home_and_Kitchen": "home",
}


def dataset_tag(category_name: str, dataset_name: str) -> str:
    """Short tag for W&B / run labels (prefer category when present)."""
    raw = (category_name or dataset_name or "unknown").strip()
    if raw in _CATEGORY_TAG_ALIASES:
        return _CATEGORY_TAG_ALIASES[raw]
    return raw.lower().replace(" ", "_")


def alias_metrics(metrics: Mapping[str, Any], *, prefix: str = "") -> dict[str, float]:
    """Map internal recall@K/ndcg@K to paper-style h@K/n@K for W&B.

    Leave-one-out recall@K equals hit-rate@K on Amazon holdout splits.
    """
    out: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        name = str(key)
        if name.startswith("recall@"):
            alias = "h@" + name.split("@", 1)[1]
        elif name.startswith("ndcg@"):
            alias = "n@" + name.split("@", 1)[1]
        else:
            alias = name
        out[f"{prefix}{alias}" if prefix else alias] = numeric
        # Also keep the original key under the same prefix for debugging.
        if alias != name:
            out[f"{prefix}{name}" if prefix else name] = numeric
    return out


class TrainingLogger:
    """File-based training logger for RecBole3.0.

    Writes every run to two places at once:
      1. A timestamped per-run file under ``<output_dir>/logs/`` (full history).
      2. A canonical, browsable copy at ``log/<model>/<dataset>.log`` so all
         models tried are visible side by side, each holding that model's run
         on the particular dataset. The canonical copy is overwritten on re-run.

    Optionally mirrors train/valid/test metrics to Weights & Biases.
    """

    def __init__(
        self,
        output_dir: str | Path,
        model_name: str,
        dataset_name: str,
        category_name: str,
        *,
        wandb_enabled: bool = False,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        wandb_mode: str = "online",
        wandb_config: Mapping[str, Any] | None = None,
        wandb_tags: Sequence[str] | None = None,
    ) -> None:
        self._is_main = _is_main_process()
        self._file = None
        self._sinks: list[Any] = []
        self._epoch_header_printed = False
        self._epoch_extra_keys: tuple[str, ...] = ()
        self._wandb = None
        self._model_name = model_name
        self._dataset_name = dataset_name
        self._category_name = category_name
        self._ds_tag = dataset_tag(category_name, dataset_name)

        if not self._is_main:
            return

        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{model_name}_{dataset_name}_{category_name}_{timestamp}.log"

        self._file = open(str(log_path), "w", encoding="utf-8")
        self._sinks.append(self._file)

        # Canonical model-organized copy: log/<model>/<dataset[_category]>.log
        run_label = category_name.strip() or dataset_name
        consolidated_dir = CONSOLIDATED_LOG_ROOT / model_name
        try:
            consolidated_dir.mkdir(parents=True, exist_ok=True)
            self._sinks.append(open(str(consolidated_dir / f"{run_label}.log"), "w", encoding="utf-8"))
        except OSError:
            # Never let logging-location problems break a training run.
            pass

        self._start_time = time_module.perf_counter()
        self._start_dt = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

        self._write_rule()
        self._writeln("  RecBole3.0 Training Log")
        self._write_rule()
        self._writeln(f"  Model: {model_name}  |  Dataset: {dataset_name}  |  Started: {self._start_dt}")
        self._write_rule()
        self._writeln()

        if wandb_enabled:
            self._init_wandb(
                output_dir=Path(output_dir),
                entity=wandb_entity,
                project=wandb_project,
                run_name=wandb_run_name,
                mode=wandb_mode,
                config=wandb_config,
                tags=wandb_tags,
            )

    def _init_wandb(
        self,
        *,
        output_dir: Path,
        entity: str | None,
        project: str | None,
        run_name: str | None,
        mode: str,
        config: Mapping[str, Any] | None,
        tags: Sequence[str] | None,
    ) -> None:
        try:
            import wandb
        except ImportError:
            self._writeln("  [wandb] package not installed; skipping W&B logging.")
            self._writeln("           Install with: pip install 'recbole3[wandb]'  (or pip install wandb)")
            self._writeln()
            return

        resolved_name = (run_name or "").strip() or output_dir.name or f"{self._model_name}_{self._ds_tag}"
        resolved_tags = list(tags) if tags else [self._ds_tag, self._model_name]
        # Keep tags unique and non-empty.
        resolved_tags = [t for i, t in enumerate(resolved_tags) if t and t not in resolved_tags[:i]]

        init_kwargs: dict[str, Any] = {
            "project": project or "textual semantic id",
            "name": resolved_name,
            "tags": resolved_tags,
            "dir": str(output_dir),
            "config": dict(config or {}),
        }
        if entity:
            init_kwargs["entity"] = entity
        if mode and mode != "online":
            init_kwargs["mode"] = mode

        try:
            self._wandb = wandb.init(**init_kwargs)
            self._writeln(
                f"  [wandb] logging to entity={entity!r} project={init_kwargs['project']!r} "
                f"run={resolved_name!r} tags={resolved_tags}"
            )
            self._writeln()
        except Exception as exc:  # noqa: BLE001 — never break training for logging
            self._wandb = None
            self._writeln(f"  [wandb] init failed ({exc!r}); continuing without W&B.")
            self._writeln()

    def _wandb_log(self, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        if self._wandb is None:
            return
        try:
            import wandb

            if step is None:
                wandb.log(dict(payload))
            else:
                wandb.log(dict(payload), step=step)
        except Exception:  # noqa: BLE001
            pass

    # ── public write helpers ──────────────────────────────────────────

    def log_config(self, title: str, config: Any) -> None:
        """Write a dataclass-based config section."""
        if not self._file:
            return
        self._writeln(f"--- Config: {title} " + "-" * (70 - len(title)))
        if dataclasses.is_dataclass(config) and not isinstance(config, type):
            tree = dataclasses.asdict(config)
            self._write_config_tree(tree, indent=0)
            if self._wandb is not None:
                try:
                    import wandb

                    wandb.config.update({title.lower(): tree}, allow_val_change=True)
                except Exception:  # noqa: BLE001
                    pass
        elif isinstance(config, Mapping):
            self._write_config_tree(dict(config), indent=0)
            if self._wandb is not None:
                try:
                    import wandb

                    wandb.config.update({title.lower(): dict(config)}, allow_val_change=True)
                except Exception:  # noqa: BLE001
                    pass
        else:
            self._writeln(f"  {config}")
        self._writeln()

    def log_model_info(self, model: Any) -> None:
        """Write model name, class, parameter counts."""
        if not self._file:
            return
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        config = getattr(model, "config", None)
        name = getattr(config, "name", "") or "unknown"
        cls_name = type(model).__name__

        self._writeln("--- Model " + "-" * 69)
        self._writeln(f"  Name:         {name}")
        self._writeln(f"  Class:        {cls_name}")
        self._writeln(f"  Params:       {total:,}")
        self._writeln(f"  Trainable:    {trainable:,}")
        self._writeln()
        self._wandb_log({"model/params": total, "model/trainable_params": trainable})

    def log_dataset_info(self, prepared_data: Any) -> None:
        """Write dataset name, task, and split sizes."""
        if not self._file:
            return
        config = getattr(prepared_data, "config", None)
        ds_name = getattr(config, "name", "") or "unknown"
        task = getattr(prepared_data, "task", "unknown")
        num_users = prepared_data.get_num_users()
        num_items = prepared_data.get_num_items()
        train_n = len(prepared_data.get_train_dataset())
        valid_n = len(prepared_data.get_eval_dataset("valid"))
        test_n = len(prepared_data.get_eval_dataset("test"))

        self._writeln("--- Dataset " + "-" * 67)
        self._writeln(f"  Name:         {ds_name}")
        self._writeln(f"  Task:         {task}")
        self._writeln(f"  Users:        {num_users:,}")
        self._writeln(f"  Items:        {num_items:,}")
        self._writeln(f"  Train:        {train_n:,} records")
        self._writeln(f"  Valid:        {valid_n:,} records")
        self._writeln(f"  Test:         {test_n:,} records")
        self._writeln()
        self._wandb_log(
            {
                "data/num_users": num_users,
                "data/num_items": num_items,
                "data/train_size": train_n,
                "data/valid_size": valid_n,
                "data/test_size": test_n,
            }
        )

    def log_epoch(
        self,
        epoch: int,
        max_epochs: int,
        loss: float | None,
        num_batches: int,
        elapsed_seconds: float,
        lr: float | None = None,
        **extra: float | None,
    ) -> None:
        """Write one row to the epoch table."""
        if not self._file:
            return

        extra_keys = tuple(extra.keys())
        if not self._epoch_header_printed:
            self._epoch_header_printed = True
            self._epoch_extra_keys = extra_keys
            self._write_epoch_header(extra_keys)

        loss_str = f"{loss:.6f}" if loss is not None else "-"
        lr_str = f"{lr:.6f}" if lr is not None else "-"
        time_str = f"{elapsed_seconds:.2f}"

        row = (
            f"  {epoch:>5d}/{max_epochs:<5d}  "
            f"{loss_str:<12s}  "
            f"{time_str:<10s}  "
            f"{num_batches:<8d}  "
            f"{lr_str:<10s}"
        )
        for key in self._epoch_extra_keys:
            val = extra.get(key)
            val = f"{val:.6f}" if isinstance(val, float) else str(val) if val is not None else "-"
            row += f"  {val:<14}"
        self._writeln(row)

        payload: dict[str, Any] = {"epoch": epoch, "train/epoch_time": elapsed_seconds, "train/num_batches": num_batches}
        if loss is not None:
            payload["train/loss"] = loss
        if lr is not None:
            payload["train/lr"] = lr
        for key in self._epoch_extra_keys:
            val = extra.get(key)
            if isinstance(val, (int, float)):
                payload[f"train/{key}"] = float(val)
        self._wandb_log(payload, step=epoch)

    def log_validation(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        """Write validation metrics for an epoch."""
        if not self._file:
            return
        formatted = self._format_metrics_line(metrics)
        self._writeln(f"    Valid epoch {epoch}: {formatted}")
        self._wandb_log(alias_metrics(metrics, prefix="valid/"), step=epoch)

    def log_best(self, epoch: int, monitor_name: str, best_value: float) -> None:
        """Record the best-epoch marker."""
        if not self._file:
            return
        self._writeln(f"\n  Best: epoch {epoch}  {monitor_name}={best_value:.6f}")
        payload = {"best/epoch": epoch, "best/value": best_value, "best/monitor": monitor_name}
        # Also alias monitor into h@/n@ when possible.
        payload.update(alias_metrics({monitor_name: best_value}, prefix="best/"))
        self._wandb_log(payload, step=epoch)

    def log_early_stopping(self, stopped: bool, epoch: int, patience: int) -> None:
        if not self._file:
            return
        if stopped:
            self._writeln(f"\n  Early stopping triggered at epoch {epoch} (patience={patience}).")
        else:
            self._writeln(f"\n  Training completed (max_epochs reached).")

    def log_test(self, test_result: Mapping[str, Any]) -> None:
        """Write final test results."""
        if not self._file:
            return
        self._writeln()
        self._write_rule()
        self._writeln("  TEST RESULTS")
        self._write_rule()
        protocol = test_result.get("protocol", "unknown")
        metrics = test_result.get("metrics", {})
        self._writeln(f"  Protocol:     {protocol}")
        if isinstance(metrics, Mapping):
            for key, value in metrics.items():
                try:
                    self._writeln(f"  {key}:{' ' * (13 - len(key))}{float(value):.6f}")
                except (TypeError, ValueError):
                    self._writeln(f"  {key}:{' ' * (13 - len(key))}{value}")
            self._wandb_log(alias_metrics(metrics, prefix="test/"))
        self._writeln()

    def log_summary(
        self,
        stopped_early: bool,
        total_epochs: int,
        best_epoch: int | None,
        total_time: float,
    ) -> None:
        """Write closing summary."""
        if not self._file:
            return
        self._writeln(f"  Total epochs:   {total_epochs}")
        self._writeln(f"  Best epoch:     {best_epoch if best_epoch is not None else 'N/A'}")
        self._writeln(f"  Stopped early:  {stopped_early}")
        self._wandb_log(
            {
                "summary/total_epochs": total_epochs,
                "summary/best_epoch": best_epoch if best_epoch is not None else -1,
                "summary/stopped_early": int(bool(stopped_early)),
                "summary/total_time_sec": total_time,
            }
        )

    def close(self) -> None:
        """Write footer and close the log files."""
        if not self._file:
            return
        elapsed = time_module.perf_counter() - self._start_time
        end_dt = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        self._writeln()
        self._write_rule()
        self._writeln(f"  Run completed: {end_dt}  (total: {elapsed:.2f} s)")
        self._write_rule()
        for sink in self._sinks:
            sink.close()
        self._sinks = []
        self._file = None
        if self._wandb is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:  # noqa: BLE001
                pass
            self._wandb = None

    # ── internal helpers ─────────────────────────────────────────────

    def _writeln(self, text: str = "") -> None:
        for sink in self._sinks:
            sink.write(text + "\n")
            sink.flush()

    def _write_rule(self) -> None:
        self._writeln("=" * 80)

    def _write_epoch_header(self, extra_keys: tuple[str, ...]) -> None:
        self._writeln()
        self._writeln("--- Epochs " + "-" * 69)
        header = "  Epoch      Avg Loss      Time (s)    Steps      LR        "
        for key in extra_keys:
            header += f"  {key:<14}"
        self._writeln(header)
        sep = "  " + "-" * (len(header) - 2)
        self._writeln(sep)

    def _write_config_tree(self, obj: Any, indent: int) -> None:
        prefix = "  " * (indent + 1)
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                if _is_nested_container(value):
                    self._writeln(f"{prefix}{key}:")
                    self._write_config_tree(value, indent + 1)
                else:
                    self._writeln(f"{prefix}{key}: {_format_scalar(value)}")
        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            for item in obj:
                if isinstance(item, Mapping):
                    self._writeln(f"{prefix}-")
                    self._write_config_tree(item, indent + 1)
                else:
                    self._writeln(f"{prefix}- {_format_scalar(item)}")
        else:
            self._writeln(f"{prefix}{_format_scalar(obj)}")

    @staticmethod
    def _format_metrics_line(metrics: Mapping[str, Any]) -> str:
        if not metrics:
            return "{}"
        parts: list[str] = []
        for name, value in metrics.items():
            try:
                parts.append(f"{name}={float(value):.6f}")
            except (TypeError, ValueError):
                parts.append(f"{name}={value}")
        return "  ".join(parts)


def _is_nested_container(obj: Any) -> bool:
    return isinstance(obj, Mapping) or (isinstance(obj, Sequence) and not isinstance(obj, str))


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, type):
        return value.__name__
    return str(value)
