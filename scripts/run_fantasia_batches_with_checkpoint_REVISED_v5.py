#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_text_exact(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text_exact(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def quote_yaml_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def replace_single_yaml_key_value_preserving_rest(
    path: Path,
    key: str,
    new_value: str,
    *,
    occurrence: int = 1,
) -> Tuple[str, str]:
    """
    Sustituye SOLO el valor de una clave YAML simple, preservando el resto del archivo.
    Devuelve (old_value_raw, new_value_quoted).
    """
    if occurrence < 1:
        raise ValueError("occurrence debe ser >= 1")

    original = read_text_exact(path)
    pattern = re.compile(
        rf"^(\s*{re.escape(key)}\s*:\s*)([^#\n]*?)(\s*(?:#.*)?)$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(original))

    if not matches:
        raise ValueError(f"No se encontró la clave YAML '{key}' en {path}. No se modifica el config.")
    if occurrence > len(matches):
        raise ValueError(
            f"Se pidió occurrence={occurrence}, pero la clave '{key}' solo aparece {len(matches)} vez/veces en {path}."
        )
    if len(matches) > 1 and occurrence == 1:
        line_numbers = [original.count("\n", 0, m.start()) + 1 for m in matches]
        raise ValueError(
            f"La clave '{key}' aparece {len(matches)} veces en {path}, líneas {line_numbers}. "
            "Por seguridad no se modifica el config. Usa --*-occurrence N si quieres seleccionar una aparición concreta."
        )

    match = matches[occurrence - 1]
    old_value_raw = match.group(2)
    new_value_quoted = quote_yaml_scalar(new_value)
    replacement = f"{match.group(1)}{new_value_quoted}{match.group(3) or ''}"
    updated = original[:match.start()] + replacement + original[match.end():]

    if original[:match.start()] != updated[:match.start()]:
        raise RuntimeError("Sanity-check falló antes de la sustitución")
    if original[match.end():] != updated[match.start() + len(replacement):]:
        raise RuntimeError("Sanity-check falló después de la sustitución")

    write_text_exact(path, updated)
    return old_value_raw, new_value_quoted


def read_yaml_scalar_raw(path: Path, key: str, *, occurrence: int = 1) -> str:
    text = read_text_exact(path)
    pattern = re.compile(
        rf"^(\s*{re.escape(key)}\s*:\s*)([^#\n]*?)(\s*(?:#.*)?)$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"No se encontró la clave YAML '{key}' en {path}.")
    if occurrence > len(matches):
        raise ValueError(f"occurrence={occurrence} supera las apariciones de '{key}' en {path}.")
    if len(matches) > 1 and occurrence == 1:
        line_numbers = [text.count("\n", 0, m.start()) + 1 for m in matches]
        raise ValueError(
            f"La clave '{key}' aparece varias veces en líneas {line_numbers}. Usa occurrence explícito."
        )
    return matches[occurrence - 1].group(2).strip()


def unquote_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        inner = value[1:-1]
        if value[0] == '"':
            inner = inner.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        return inner
    return value


def build_batch_prefix(base_prefix_raw: str, batch_label: str, batch_path: Path, mode: str) -> str:
    base_prefix = unquote_yaml_scalar(base_prefix_raw)
    if mode == "suffix_label":
        suffix = batch_label
    elif mode == "suffix_stem":
        suffix = batch_path.stem
    else:
        raise ValueError(f"Modo de prefix no soportado: {mode}")
    return f"{base_prefix}_{suffix}"


def read_state(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "current_stage": "INIT",
        "current_batch": None,
        "batches": {},
        "history": [],
    }


def write_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


def discover_batches(batches_dir: Path, batch_pattern: str) -> List[Path]:
    if not batches_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de batches: {batches_dir}")
    if not batches_dir.is_dir():
        raise NotADirectoryError(f"La ruta de batches no es un directorio: {batches_dir}")
    batches = sorted(p.resolve() for p in batches_dir.glob(batch_pattern) if p.is_file())
    if not batches:
        raise FileNotFoundError(
            f"No se encontraron batches en {batches_dir} con el patrón '{batch_pattern}'."
        )
    return batches


def resolve_selector_token(part: str, total: int) -> int:
    value = part.strip().lower()
    if value == "first":
        return 1
    if value == "last":
        return total
    idx = int(value)
    if idx < 1 or idx > total:
        raise ValueError(f"Índice de batch fuera de rango: {idx}. Rango válido: 1..{total}")
    return idx


def parse_batch_select(selector: str, total: int) -> List[int]:
    selector = selector.strip().lower()
    if not selector or selector == "all":
        return list(range(1, total + 1))

    selected: List[int] = []
    seen = set()

    for raw_token in selector.split(','):
        token = raw_token.strip()
        if not token:
            continue

        if token == "all":
            indices = list(range(1, total + 1))
        elif '-' in token:
            left, right = token.split('-', 1)
            start = resolve_selector_token(left, total)
            end = resolve_selector_token(right, total)
            step = 1 if start <= end else -1
            indices = list(range(start, end + step, step))
        else:
            indices = [resolve_selector_token(token, total)]

        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)

    if not selected:
        raise ValueError("La selección de batches está vacía.")

    return selected


def load_args_from_txt(args_file: Path) -> List[str]:
    if not args_file.exists():
        raise FileNotFoundError(f"No existe el archivo de argumentos: {args_file}")
    text = args_file.read_text(encoding="utf-8")
    # Acepta espacios, saltos de línea, comillas y comentarios con #
    return shlex.split(text, comments=True, posix=True)


class BatchRunner:
    def __init__(
        self,
        config_path: Path,
        batches_dir: Path,
        batch_pattern: str,
        batch_select: str,
        checkpoint_path: Path,
        logs_dir: Path,
        runner_cmd: str,
        input_key: str,
        input_key_occurrence: int,
        prefix_key: Optional[str],
        prefix_key_occurrence: int,
        prefix_mode: str,
        dry_run: bool,
        restore_config: bool,
        max_batches: Optional[int],
    ) -> None:
        self.config_path = config_path.resolve()
        self.batches_dir = batches_dir.resolve()
        self.batch_pattern = batch_pattern
        self.batch_select = batch_select
        self.checkpoint_path = checkpoint_path.resolve()
        self.logs_dir = logs_dir.resolve()
        self.runner_cmd = runner_cmd
        self.input_key = input_key
        self.input_key_occurrence = input_key_occurrence
        self.prefix_key = prefix_key
        self.prefix_key_occurrence = prefix_key_occurrence
        self.prefix_mode = prefix_mode
        self.dry_run = dry_run
        self.restore_config = restore_config
        self.max_batches = max_batches
        self.backup_path = self.config_path.with_suffix(self.config_path.suffix + ".bak_before_batches")

    def ensure_backup(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"No existe el config: {self.config_path}")
        if not self.backup_path.exists():
            shutil.copy2(self.config_path, self.backup_path)

    def restore_backup(self) -> None:
        if self.backup_path.exists():
            shutil.copy2(self.backup_path, self.config_path)

    def _prepare_batch_config(self, batch_path: Path, batch_label: str) -> Dict[str, Any]:
        change_info: Dict[str, Any] = {}

        old_input_raw, new_input = replace_single_yaml_key_value_preserving_rest(
            self.config_path,
            self.input_key,
            str(batch_path),
            occurrence=self.input_key_occurrence,
        )
        change_info["input"] = {
            "changed_key": self.input_key,
            "changed_key_occurrence": self.input_key_occurrence,
            "old_value_raw": old_input_raw,
            "new_value": new_input,
        }

        if self.prefix_key:
            # Importante: el prefix base se toma SIEMPRE del backup original,
            # para evitar concatenaciones acumulativas entre batches.
            prefix_source = self.backup_path if self.backup_path.exists() else self.config_path
            original_prefix_raw = read_yaml_scalar_raw(
                prefix_source,
                self.prefix_key,
                occurrence=self.prefix_key_occurrence,
            )
            new_prefix_value = build_batch_prefix(original_prefix_raw, batch_label, batch_path, self.prefix_mode)
            old_prefix_raw, new_prefix = replace_single_yaml_key_value_preserving_rest(
                self.config_path,
                self.prefix_key,
                new_prefix_value,
                occurrence=self.prefix_key_occurrence,
            )
            change_info["prefix"] = {
                "changed_key": self.prefix_key,
                "changed_key_occurrence": self.prefix_key_occurrence,
                "old_value_raw": old_prefix_raw,
                "base_prefix_raw": original_prefix_raw,
                "new_value": new_prefix,
                "mode": self.prefix_mode,
            }

        return change_info

    def _runner_log_file(self, batch_label: str) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self.logs_dir / f"fantasia_runner_{batch_label}.log"

    def _build_cmd(self) -> List[str]:
        return shlex.split(self.runner_cmd) + ["--config", str(self.config_path)]

    def _make_plan(self) -> List[Tuple[str, Path, int]]:
        discovered = discover_batches(self.batches_dir, self.batch_pattern)
        if self.max_batches is not None:
            discovered = discovered[: self.max_batches]

        selected_positions = parse_batch_select(self.batch_select, len(discovered))
        plan: List[Tuple[str, Path, int]] = []
        for pos in selected_positions:
            batch_path = discovered[pos - 1]
            label = f"batch_{pos:05d}"
            plan.append((label, batch_path, pos))
        return plan

    def run_batch(self, state: Dict[str, Any], batch_label: str, batch_path: Path, source_position: int) -> int:
        self.ensure_backup()
        if not batch_path.exists():
            raise FileNotFoundError(f"No existe el FASTA del {batch_label}: {batch_path}")

        runner_log = self._runner_log_file(batch_label)
        state["current_stage"] = f"PREPARE_{batch_label.upper()}"
        state["current_batch"] = batch_label
        state.setdefault("batches", {}).setdefault(batch_label, {})
        state["batches"][batch_label].update({
            "input": str(batch_path),
            "batch_file_name": batch_path.name,
            "source_position": source_position,
            "prepare_started_at": now_iso(),
            "status": "preparing",
        })
        write_state(self.checkpoint_path, state)

        change_info = self._prepare_batch_config(batch_path, batch_label)

        state["batches"][batch_label].update({
            "prepare_finished_at": now_iso(),
            "config_change": change_info,
            "effective_config_path": str(self.config_path),
            "status": "running",
            "run_started_at": now_iso(),
        })
        state["current_stage"] = f"RUNNING_{batch_label.upper()}"
        state["history"].append({
            "event": "batch_start",
            "batch": batch_label,
            "batch_file_name": batch_path.name,
            "source_position": source_position,
            "at": now_iso(),
            "input": str(batch_path),
            "changed_keys": list(change_info.keys()),
        })
        write_state(self.checkpoint_path, state)

        cmd = self._build_cmd()
        state["batches"][batch_label]["command"] = cmd
        state["batches"][batch_label]["runner_log_file"] = str(runner_log)
        write_state(self.checkpoint_path, state)

        if self.dry_run:
            with runner_log.open("a", encoding="utf-8") as log:
                log.write(f"[{now_iso()}] DRY RUN: {' '.join(shlex.quote(x) for x in cmd)}\n")
            rc = 0
        else:
            with runner_log.open("a", encoding="utf-8") as log:
                log.write(f"[{now_iso()}] START {' '.join(shlex.quote(x) for x in cmd)}\n")
                log.flush()
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
                rc = proc.returncode
                log.write(f"[{now_iso()}] END returncode={rc}\n")

        state["batches"][batch_label].update({
            "run_finished_at": now_iso(),
            "return_code": rc,
        })

        if rc == 0:
            state["batches"][batch_label]["status"] = "done"
            state["history"].append({"event": "batch_done", "batch": batch_label, "at": now_iso()})
        else:
            state["batches"][batch_label]["status"] = "failed"
            state["current_stage"] = f"FAILED_{batch_label.upper()}"
            state["history"].append({
                "event": "batch_failed",
                "batch": batch_label,
                "at": now_iso(),
                "return_code": rc,
            })
        write_state(self.checkpoint_path, state)
        return rc

    def run(self) -> int:
        state = read_state(self.checkpoint_path)
        self.ensure_backup()
        plan = self._make_plan()
        state["plan"] = [
            {
                "batch_label": batch_label,
                "batch_path": str(batch_path),
                "batch_file_name": batch_path.name,
                "source_position": source_position,
            }
            for batch_label, batch_path, source_position in plan
        ]
        write_state(self.checkpoint_path, state)

        try:
            for batch_label, batch_path, source_position in plan:
                batch_state = state.get("batches", {}).get(batch_label, {})
                if batch_state.get("status") == "done":
                    continue
                rc = self.run_batch(state, batch_label, batch_path, source_position)
                if rc != 0:
                    return rc
            state["current_stage"] = "ALL_DONE"
            state["current_batch"] = None
            state["finished_at"] = now_iso()
            state["history"].append({"event": "all_done", "at": now_iso()})
            write_state(self.checkpoint_path, state)
            return 0
        except KeyboardInterrupt:
            state["current_stage"] = "INTERRUPTED"
            state["history"].append({
                "event": "keyboard_interrupt",
                "at": now_iso(),
                "current_batch": state.get("current_batch"),
            })
            write_state(self.checkpoint_path, state)
            return 130
        finally:
            if self.restore_config:
                self.restore_backup()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Ejecuta FANTASIA sobre batches de un directorio, de forma secuencial, modificando "
            "temporalmente solo las claves de input y prefix del config.yaml y preservando el resto "
            "del fichero byte a byte."
        )
    )
    ap.add_argument(
        "--args-file",
        default=None,
        help=(
            "Ruta a un TXT con argumentos para el script. El archivo puede contener los mismos flags "
            "que pondrías en la terminal, con espacios, saltos de línea, comillas y comentarios #."
        ),
    )
    ap.add_argument("--config", required=True, help="Ruta al config.yaml de FANTASIA")
    ap.add_argument("--batches-dir", required=True, help="Directorio con los batches FASTA/.fa a procesar")
    ap.add_argument(
        "--batch-pattern",
        default="*.fa*",
        help="Patrón glob para descubrir batches dentro de --batches-dir. Por defecto: *.fa*",
    )
    ap.add_argument(
        "--batch-select",
        default="all",
        help=(
            "Qué batches procesar, usando el orden descubierto en la carpeta. Ejemplos: all, 1, last, 1,last, 1,2, 1-5, 3-last."
        ),
    )
    ap.add_argument("--checkpoint", default="./fantasia_batches_checkpoint.json", help="Ruta al JSON de checkpoint/estado")
    ap.add_argument("--logs-dir", default="./fantasia_batch_logs", help="Directorio donde se guardará el log del runner")
    ap.add_argument("--runner-cmd", default="poetry run fantasia run", help="Comando base. Por defecto: 'poetry run fantasia run'.")

    ap.add_argument("--input-key", default="input", help="Clave YAML del FASTA de entrada. Por defecto: input")
    ap.add_argument("--input-key-occurrence", type=int, default=1, help="Aparición de --input-key a modificar si la clave existe varias veces.")

    ap.add_argument("--prefix-key", default="prefix", help="Clave YAML del prefijo de salida. Usa '' para no tocar el prefix.")
    ap.add_argument("--prefix-key-occurrence", type=int, default=1, help="Aparición de --prefix-key a modificar si la clave existe varias veces.")
    ap.add_argument(
        "--prefix-mode",
        choices=["suffix_label", "suffix_stem"],
        default="suffix_stem",
        help=(
            "Cómo construir el nuevo prefix. suffix_label => <prefix_original>_batch_00001. "
            "suffix_stem => <prefix_original>_<stem_del_fasta>. Por defecto: suffix_stem."
        ),
    )

    ap.add_argument("--max-batches", type=int, default=None, help="Recorta el conjunto descubierto a los primeros N batches antes de aplicar --batch-select")
    ap.add_argument("--dry-run", action="store_true", help="Prepara config/checkpoint pero no ejecuta FANTASIA")
    ap.add_argument("--no-restore-config", action="store_true", help="No restaura el config original al final")
    return ap


def expand_args_file(argv: Sequence[str]) -> List[str]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--args-file")
    known, remaining = pre.parse_known_args(list(argv))
    if not known.args_file:
        return list(argv)

    file_tokens = load_args_from_txt(Path(known.args_file).resolve())
    # Los argumentos de CLI que acompañen a --args-file tienen prioridad si repiten flags.
    return file_tokens + remaining


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    final_argv = expand_args_file(raw_argv)
    args = build_parser().parse_args(final_argv)

    prefix_key = args.prefix_key if args.prefix_key != "" else None
    runner = BatchRunner(
        config_path=Path(args.config),
        batches_dir=Path(args.batches_dir),
        batch_pattern=args.batch_pattern,
        batch_select=args.batch_select,
        checkpoint_path=Path(args.checkpoint),
        logs_dir=Path(args.logs_dir),
        runner_cmd=args.runner_cmd,
        input_key=args.input_key,
        input_key_occurrence=args.input_key_occurrence,
        prefix_key=prefix_key,
        prefix_key_occurrence=args.prefix_key_occurrence,
        prefix_mode=args.prefix_mode,
        dry_run=args.dry_run,
        restore_config=not args.no_restore_config,
        max_batches=args.max_batches,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
