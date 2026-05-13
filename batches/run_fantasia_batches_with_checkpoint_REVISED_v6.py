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

try:
    import yaml
except ImportError as exc:
    raise SystemExit("ERROR: Este script requiere PyYAML (pip install pyyaml)") from exc


# -----------------------------------------------------------------------------
# Helpers generales
# -----------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def now_compact() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d%H%M%S")


def read_text_exact(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text_exact(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def quote_yaml_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def unquote_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        inner = value[1:-1]
        if value[0] == '"':
            inner = inner.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        return inner
    return value


def expand_path(value: str) -> Path:
    return Path(unquote_yaml_scalar(value)).expanduser().resolve()


def sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())


# -----------------------------------------------------------------------------
# Sustituciones YAML preservando el resto del fichero
# -----------------------------------------------------------------------------

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


def replace_model_enabled_preserving_rest(path: Path, model_name: str, enabled: bool) -> Tuple[str, str]:
    """
    Modifica SOLO la línea 'enabled:' dentro del bloque del modelo indicado,
    preservando el resto del YAML exactamente igual.
    Devuelve (old_value_raw, new_value_raw).
    """
    lines = read_text_exact(path).splitlines(keepends=True)
    model_line_idx: Optional[int] = None
    model_indent: Optional[int] = None

    # Localizar la línea exacta del modelo, por ejemplo: '    Prot-T5:'
    model_re = re.compile(rf"^(?P<indent>\s*){re.escape(model_name)}\s*:\s*(#.*)?$")
    for i, line in enumerate(lines):
        m = model_re.match(line.rstrip("\n"))
        if m:
            model_line_idx = i
            model_indent = len(m.group("indent"))
            break

    if model_line_idx is None or model_indent is None:
        raise ValueError(f"No se encontró el bloque del modelo '{model_name}' en {path}.")

    enabled_idx: Optional[int] = None
    old_value_raw: Optional[str] = None

    enabled_re = re.compile(r"^(?P<prefix>\s*enabled\s*:\s*)(?P<value>[^#\n]*?)(?P<suffix>\s*(?:#.*)?)$")

    for j in range(model_line_idx + 1, len(lines)):
        raw = lines[j]
        stripped = raw.strip()
        if not stripped:
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        # Fin del bloque del modelo cuando volvemos a una indentación <= a la del modelo.
        if indent <= model_indent and not stripped.startswith("#"):
            break

        m_enabled = enabled_re.match(raw.rstrip("\n"))
        if m_enabled:
            enabled_idx = j
            old_value_raw = m_enabled.group("value").strip()
            new_value_raw = "True" if enabled else "False"
            lines[j] = f"{m_enabled.group('prefix')}{new_value_raw}{m_enabled.group('suffix')}\n"
            break

    if enabled_idx is None or old_value_raw is None:
        raise ValueError(f"No se encontró la clave 'enabled' dentro del modelo '{model_name}' en {path}.")

    write_text_exact(path, "".join(lines))
    return old_value_raw, ("True" if enabled else "False")


# -----------------------------------------------------------------------------
# Metadata del config y estado
# -----------------------------------------------------------------------------

def load_config_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"El config YAML no tiene un objeto raíz tipo dict: {path}")
    return data


def read_state(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "current_stage": "INIT",
        "current_model": None,
        "current_batch": None,
        "runs": {},
        "history": [],
    }


def write_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


# -----------------------------------------------------------------------------
# Selección de batches/modelos y args-file
# -----------------------------------------------------------------------------

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


def parse_model_select(selector: str, available_models: List[str], enabled_models: List[str]) -> List[str]:
    if not selector or selector.strip().lower() == "all":
        return available_models
    if selector.strip().lower() == "enabled":
        return enabled_models

    requested = [x.strip() for x in selector.split(',') if x.strip()]
    unknown = [x for x in requested if x not in available_models]
    if unknown:
        raise ValueError(
            f"Modelos no reconocidos en --model-select: {unknown}. Disponibles: {available_models}"
        )
    return requested


def load_args_from_txt(args_file: Path) -> List[str]:
    if not args_file.exists():
        raise FileNotFoundError(f"No existe el archivo de argumentos: {args_file}")
    text = args_file.read_text(encoding="utf-8")
    return shlex.split(text, comments=True, posix=True)


def expand_args_file(argv: Sequence[str]) -> List[str]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--args-file")
    known, remaining = pre.parse_known_args(list(argv))
    if not known.args_file:
        return list(argv)
    file_tokens = load_args_from_txt(Path(known.args_file).resolve())
    return file_tokens + remaining


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

class ModelBatchRunner:
    def __init__(
        self,
        config_path: Path,
        batches_dir: Path,
        batch_pattern: str,
        batch_select: str,
        model_select: str,
        checkpoint_path: Path,
        runner_logs_dir: Path,
        runner_cmd: str,
        input_key: str,
        input_key_occurrence: int,
        prefix_key: Optional[str],
        prefix_key_occurrence: int,
        log_path_key: Optional[str],
        log_path_key_occurrence: int,
        dry_run: bool,
        restore_config: bool,
        max_batches: Optional[int],
        session_name_base: Optional[str],
    ) -> None:
        self.config_path = config_path.resolve()
        self.batches_dir = batches_dir.resolve()
        self.batch_pattern = batch_pattern
        self.batch_select = batch_select
        self.model_select = model_select
        self.checkpoint_path = checkpoint_path.resolve()
        self.runner_logs_dir = runner_logs_dir.resolve()
        self.runner_cmd = runner_cmd
        self.input_key = input_key
        self.input_key_occurrence = input_key_occurrence
        self.prefix_key = prefix_key
        self.prefix_key_occurrence = prefix_key_occurrence
        self.log_path_key = log_path_key
        self.log_path_key_occurrence = log_path_key_occurrence
        self.dry_run = dry_run
        self.restore_config = restore_config
        self.max_batches = max_batches
        self.backup_path = self.config_path.with_suffix(self.config_path.suffix + ".bak_before_batches")

        self.ensure_backup()
        self.original_meta = load_config_metadata(self.backup_path)
        self.available_models = list((self.original_meta.get("models") or {}).keys())
        self.enabled_models = [
            name for name, spec in (self.original_meta.get("models") or {}).items()
            if isinstance(spec, dict) and bool(spec.get("enabled"))
        ]
        if not self.available_models:
            raise ValueError(f"No se encontraron modelos bajo la clave 'models' en {self.config_path}")

        base_dir_raw = self.original_meta.get("base_directory")
        if not isinstance(base_dir_raw, str):
            raise ValueError("El config no contiene 'base_directory' como string")
        self.base_directory = expand_path(base_dir_raw)
        self.experiments_root = (self.base_directory / "experiments").resolve()

        log_path_raw = self.original_meta.get("log_path")
        if not isinstance(log_path_raw, str):
            raise ValueError("El config no contiene 'log_path' como string")
        self.logs_root = expand_path(log_path_raw)

        self.models_plan = parse_model_select(self.model_select, self.available_models, self.enabled_models)
        if not self.models_plan:
            raise ValueError("La selección de modelos está vacía.")

        self.session_timestamp = now_compact()
        base_name = session_name_base.strip() if session_name_base else self.batches_dir.name
        self.session_name = f"{sanitize_label(base_name)}_{self.session_timestamp}"

    def ensure_backup(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"No existe el config: {self.config_path}")
        if not self.backup_path.exists():
            shutil.copy2(self.config_path, self.backup_path)

    def restore_backup(self) -> None:
        if self.backup_path.exists():
            shutil.copy2(self.backup_path, self.config_path)

    def _set_models_enabled(self, selected_model: str) -> Dict[str, Dict[str, str]]:
        changes: Dict[str, Dict[str, str]] = {}
        for model_name in self.available_models:
            desired = (model_name == selected_model)
            old_raw, new_raw = replace_model_enabled_preserving_rest(self.config_path, model_name, desired)
            changes[model_name] = {
                "old_enabled": old_raw,
                "new_enabled": new_raw,
            }
        return changes

    def _prepare_run_config(
        self,
        model_name: str,
        batch_path: Path,
        batch_label: str,
        internal_prefix: str,
        final_logs_batch_root: Path,
    ) -> Dict[str, Any]:
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
            old_prefix_raw, new_prefix = replace_single_yaml_key_value_preserving_rest(
                self.config_path,
                self.prefix_key,
                internal_prefix,
                occurrence=self.prefix_key_occurrence,
            )
            change_info["prefix"] = {
                "changed_key": self.prefix_key,
                "changed_key_occurrence": self.prefix_key_occurrence,
                "old_value_raw": old_prefix_raw,
                "new_value": new_prefix,
            }

        if self.log_path_key:
            final_logs_batch_root.mkdir(parents=True, exist_ok=True)
            old_log_raw, new_log = replace_single_yaml_key_value_preserving_rest(
                self.config_path,
                self.log_path_key,
                str(final_logs_batch_root),
                occurrence=self.log_path_key_occurrence,
            )
            change_info["log_path"] = {
                "changed_key": self.log_path_key,
                "changed_key_occurrence": self.log_path_key_occurrence,
                "old_value_raw": old_log_raw,
                "new_value": new_log,
            }

        change_info["models_enabled"] = self._set_models_enabled(model_name)
        return change_info

    def _runner_log_file(self, run_label: str) -> Path:
        self.runner_logs_dir.mkdir(parents=True, exist_ok=True)
        return self.runner_logs_dir / f"fantasia_runner_{sanitize_label(run_label)}.log"

    def _build_cmd(self) -> List[str]:
        return shlex.split(self.runner_cmd) + ["--config", str(self.config_path)]

    def _make_plan(self) -> List[Dict[str, Any]]:
        discovered = discover_batches(self.batches_dir, self.batch_pattern)
        if self.max_batches is not None:
            discovered = discovered[: self.max_batches]

        selected_positions = parse_batch_select(self.batch_select, len(discovered))
        selected_batches = [(pos, discovered[pos - 1]) for pos in selected_positions]

        plan: List[Dict[str, Any]] = []
        for model_name in self.models_plan:
            model_dir_name = model_name
            for pos, batch_path in selected_batches:
                batch_label = f"batch_{pos:05d}"
                batch_stem = batch_path.stem
                run_stamp = now_compact()
                final_batch_dir_name = f"{batch_stem}_{run_stamp}"
                internal_prefix = f"{sanitize_label(model_name)}_{final_batch_dir_name}"
                run_label = f"{model_name}__{batch_label}"
                final_experiment_dir = self.experiments_root / self.session_name / model_dir_name / final_batch_dir_name
                final_logs_batch_root = self.logs_root / self.session_name / model_dir_name / final_batch_dir_name
                plan.append(
                    {
                        "run_label": run_label,
                        "model_name": model_name,
                        "model_dir_name": model_dir_name,
                        "batch_label": batch_label,
                        "source_position": pos,
                        "batch_path": batch_path,
                        "batch_file_name": batch_path.name,
                        "batch_stem": batch_stem,
                        "run_stamp": run_stamp,
                        "final_batch_dir_name": final_batch_dir_name,
                        "internal_prefix": internal_prefix,
                        "final_experiment_dir": str(final_experiment_dir),
                        "final_logs_batch_root": str(final_logs_batch_root),
                    }
                )
        return plan

    def _find_created_experiment_dir(self, internal_prefix: str) -> Optional[Path]:
        candidates = [p for p in self.experiments_root.glob(f"{internal_prefix}*") if p.is_dir()]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def run_single(self, state: Dict[str, Any], run: Dict[str, Any]) -> int:
        run_label = run["run_label"]
        model_name = run["model_name"]
        batch_label = run["batch_label"]
        batch_path = Path(run["batch_path"]).resolve()
        source_position = int(run["source_position"])
        internal_prefix = run["internal_prefix"]
        final_experiment_dir = Path(run["final_experiment_dir"]).resolve()
        final_logs_batch_root = Path(run["final_logs_batch_root"]).resolve()
        runner_log = self._runner_log_file(run_label)

        state["current_stage"] = f"PREPARE_{sanitize_label(model_name).upper()}_{batch_label.upper()}"
        state["current_model"] = model_name
        state["current_batch"] = batch_label
        state.setdefault("runs", {}).setdefault(run_label, {})
        state["runs"][run_label].update({
            "model_name": model_name,
            "batch_label": batch_label,
            "batch_file_name": batch_path.name,
            "batch_path": str(batch_path),
            "source_position": source_position,
            "internal_prefix": internal_prefix,
            "final_experiment_dir": str(final_experiment_dir),
            "final_logs_batch_root": str(final_logs_batch_root),
            "prepare_started_at": now_iso(),
            "status": "preparing",
        })
        write_state(self.checkpoint_path, state)

        change_info = self._prepare_run_config(
            model_name=model_name,
            batch_path=batch_path,
            batch_label=batch_label,
            internal_prefix=internal_prefix,
            final_logs_batch_root=final_logs_batch_root,
        )

        state["runs"][run_label].update({
            "prepare_finished_at": now_iso(),
            "config_change": change_info,
            "effective_config_path": str(self.config_path),
            "status": "running",
            "run_started_at": now_iso(),
        })
        state["current_stage"] = f"RUNNING_{sanitize_label(model_name).upper()}_{batch_label.upper()}"
        state["history"].append({
            "event": "run_start",
            "run_label": run_label,
            "model_name": model_name,
            "batch_label": batch_label,
            "batch_file_name": batch_path.name,
            "source_position": source_position,
            "at": now_iso(),
            "changed_keys": list(change_info.keys()),
        })
        write_state(self.checkpoint_path, state)

        cmd = self._build_cmd()
        state["runs"][run_label]["command"] = cmd
        state["runs"][run_label]["runner_log_file"] = str(runner_log)
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

        created_experiment_dir: Optional[Path] = None
        relocation_error: Optional[str] = None
        if rc == 0 and not self.dry_run:
            try:
                created_experiment_dir = self._find_created_experiment_dir(internal_prefix)
                if created_experiment_dir is not None:
                    final_experiment_dir.parent.mkdir(parents=True, exist_ok=True)
                    if final_experiment_dir.exists():
                        raise FileExistsError(
                            f"El directorio final de experimento ya existe: {final_experiment_dir}"
                        )
                    shutil.move(str(created_experiment_dir), str(final_experiment_dir))
                else:
                    relocation_error = (
                        f"No se encontró el directorio de experimento creado para prefix '{internal_prefix}' "
                        f"bajo {self.experiments_root}"
                    )
            except Exception as exc:
                relocation_error = str(exc)

        state["runs"][run_label].update({
            "run_finished_at": now_iso(),
            "return_code": rc,
            "created_experiment_dir": str(created_experiment_dir) if created_experiment_dir else None,
            "relocation_error": relocation_error,
        })

        if rc == 0:
            state["runs"][run_label]["status"] = "done"
            state["history"].append({
                "event": "run_done",
                "run_label": run_label,
                "model_name": model_name,
                "batch_label": batch_label,
                "at": now_iso(),
                "relocation_error": relocation_error,
            })
        else:
            state["runs"][run_label]["status"] = "failed"
            state["current_stage"] = f"FAILED_{sanitize_label(model_name).upper()}_{batch_label.upper()}"
            state["history"].append({
                "event": "run_failed",
                "run_label": run_label,
                "model_name": model_name,
                "batch_label": batch_label,
                "at": now_iso(),
                "return_code": rc,
            })
        write_state(self.checkpoint_path, state)
        return rc

    def run(self) -> int:
        state = read_state(self.checkpoint_path)
        plan = self._make_plan()
        state["session_name"] = self.session_name
        state["session_timestamp"] = self.session_timestamp
        state["models_plan"] = self.models_plan
        state["plan"] = plan
        write_state(self.checkpoint_path, state)

        try:
            for run in plan:
                run_label = run["run_label"]
                run_state = state.get("runs", {}).get(run_label, {})
                if run_state.get("status") == "done":
                    continue
                rc = self.run_single(state, run)
                if rc != 0:
                    return rc
            state["current_stage"] = "ALL_DONE"
            state["current_model"] = None
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
                "current_model": state.get("current_model"),
                "current_batch": state.get("current_batch"),
            })
            write_state(self.checkpoint_path, state)
            return 130
        finally:
            if self.restore_config:
                self.restore_backup()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Ejecuta FANTASIA sobre batches de un directorio y sobre múltiples modelos, de forma secuencial, "
            "modificando temporalmente solo las claves necesarias del config.yaml (input, prefix, log_path y enabled) "
            "y preservando el resto del fichero byte a byte."
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
    ap.add_argument(
        "--model-select",
        default="all",
        help=(
            "Qué modelos procesar. Valores soportados: all, enabled o una lista separada por comas, "
            "por ejemplo: ESM,Prot-T5,ESM3c"
        ),
    )
    ap.add_argument(
        "--session-name-base",
        default=None,
        help=(
            "Nombre base del experimento agrupador. Se le añadirá automáticamente un timestamp. "
            "Si no se indica, se usa el nombre del directorio de batches."
        ),
    )
    ap.add_argument("--checkpoint", default="./fantasia_batches_checkpoint.json", help="Ruta al JSON de checkpoint/estado")
    ap.add_argument("--logs-dir", default="./fantasia_batch_runner_logs", help="Directorio donde se guardará el log del runner")
    ap.add_argument("--runner-cmd", default="poetry run fantasia run", help="Comando base. Por defecto: 'poetry run fantasia run'.")

    ap.add_argument("--input-key", default="input", help="Clave YAML del FASTA de entrada. Por defecto: input")
    ap.add_argument("--input-key-occurrence", type=int, default=1, help="Aparición de --input-key a modificar si la clave existe varias veces.")

    ap.add_argument("--prefix-key", default="prefix", help="Clave YAML del prefijo de salida. Usa '' para no tocar el prefix.")
    ap.add_argument("--prefix-key-occurrence", type=int, default=1, help="Aparición de --prefix-key a modificar si la clave existe varias veces.")

    ap.add_argument("--log-path-key", default="log_path", help="Clave YAML donde se define el directorio de logs de FANTASIA. Por defecto: log_path")
    ap.add_argument("--log-path-key-occurrence", type=int, default=1, help="Aparición de --log-path-key a modificar si la clave existe varias veces.")

    ap.add_argument("--max-batches", type=int, default=None, help="Recorta el conjunto descubierto a los primeros N batches antes de aplicar --batch-select")
    ap.add_argument("--dry-run", action="store_true", help="Prepara config/checkpoint pero no ejecuta FANTASIA")
    ap.add_argument("--no-restore-config", action="store_true", help="No restaura el config original al final")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    final_argv = expand_args_file(raw_argv)
    args = build_parser().parse_args(final_argv)

    prefix_key = args.prefix_key if args.prefix_key != "" else None
    log_path_key = args.log_path_key if args.log_path_key != "" else None

    runner = ModelBatchRunner(
        config_path=Path(args.config),
        batches_dir=Path(args.batches_dir),
        batch_pattern=args.batch_pattern,
        batch_select=args.batch_select,
        model_select=args.model_select,
        checkpoint_path=Path(args.checkpoint),
        runner_logs_dir=Path(args.logs_dir),
        runner_cmd=args.runner_cmd,
        input_key=args.input_key,
        input_key_occurrence=args.input_key_occurrence,
        prefix_key=prefix_key,
        prefix_key_occurrence=args.prefix_key_occurrence,
        log_path_key=log_path_key,
        log_path_key_occurrence=args.log_path_key_occurrence,
        dry_run=args.dry_run,
        restore_config=not args.no_restore_config,
        max_batches=args.max_batches,
        session_name_base=args.session_name_base,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
