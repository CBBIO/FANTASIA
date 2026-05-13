#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import shlex
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, TextIO, Tuple


@dataclass
class BatchInfo:
    batch_index: int
    file_path: str
    n_sequences: int
    min_length: Optional[int]
    max_length: Optional[int]
    first_header: Optional[str]
    last_header: Optional[str]


@dataclass
class LengthStats:
    input_path: str
    total_sequences: int
    kept_sequences: int
    filtered_sequences: int
    min_kept_length: Optional[int]
    max_kept_length: Optional[int]
    thresholds: Dict[str, int]
    exact_lengths: Dict[str, int]
    filtered_fasta_path: Optional[str]
    filtered_manifest_path: Optional[str]


def open_text_auto(path: Path) -> TextIO:
    if path.suffix.lower() == '.gz':
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    return path.open('r', encoding='utf-8', errors='replace')


def fasta_records(handle: TextIO) -> Generator[Tuple[str, List[str]], None, None]:
    header: Optional[str] = None
    seq_lines: List[str] = []

    for raw_line in handle:
        line = raw_line.rstrip('\n')
        if not line:
            continue
        if line.startswith('>'):
            if header is not None:
                yield header, seq_lines
            header = line
            seq_lines = []
        else:
            if header is None:
                raise ValueError(
                    "El archivo no parece FASTA válido: se encontró secuencia antes de la primera cabecera '>'."
                )
            seq_lines.append(line.strip())

    if header is not None:
        yield header, seq_lines


def infer_output_suffix(input_path: Path) -> str:
    name = input_path.name.lower()
    if name.endswith('.fasta.gz'):
        return '.fasta'
    if name.endswith('.fa.gz'):
        return '.fa'
    if input_path.suffix.lower() in {'.fa', '.fasta'}:
        return input_path.suffix
    return '.fasta'


def base_stem(path: Path) -> str:
    stem = path.name
    for ending in ('.fasta.gz', '.fa.gz', '.fasta', '.fa'):
        if stem.lower().endswith(ending):
            return stem[: -len(ending)]
    return path.stem


def write_record(handle: TextIO, header: str, seq_lines: List[str]) -> None:
    handle.write(f"{header}\n")
    for line in seq_lines:
        handle.write(f"{line}\n")


def extract_identifier(header: str) -> str:
    cleaned = header[1:] if header.startswith('>') else header
    return cleaned.split()[0] if cleaned.strip() else ''


def load_args_from_txt(args_file: Path) -> List[str]:
    if not args_file.exists():
        raise FileNotFoundError(f"No existe el archivo de argumentos: {args_file}")
    text = args_file.read_text(encoding='utf-8')
    return shlex.split(text, comments=True, posix=True)


def expand_args_file(argv: Sequence[str]) -> List[str]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--args-file')
    known, remaining = pre.parse_known_args(list(argv))
    if not known.args_file:
        return list(argv)
    file_tokens = load_args_from_txt(Path(known.args_file).resolve())
    return file_tokens + remaining


def prepare_output_file(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(
                f"El archivo de salida ya existe: {path}. Usa --overwrite si quieres sobrescribir."
            )


def distribute_by_length(
    input_path: Path,
    bucket_dir: Path,
    min_length_kept: int,
    report_thresholds: List[int],
    report_only: bool,
    write_filtered_trace: bool,
    filtered_fasta_path: Optional[Path],
    filtered_manifest_path: Optional[Path],
    overwrite: bool,
) -> Tuple[LengthStats, List[int]]:
    bucket_dir.mkdir(parents=True, exist_ok=True)

    total_sequences = 0
    kept_sequences = 0
    filtered_sequences = 0
    min_kept_length: Optional[int] = None
    max_kept_length: Optional[int] = None
    threshold_counts = Counter()
    exact_length_counts = Counter()
    kept_lengths_seen = set()

    filtered_fasta_handle: Optional[TextIO] = None
    filtered_manifest_handle: Optional[TextIO] = None
    filtered_manifest_writer: Optional[csv.DictWriter] = None

    if write_filtered_trace and not report_only:
        assert filtered_fasta_path is not None
        assert filtered_manifest_path is not None
        prepare_output_file(filtered_fasta_path, overwrite=overwrite)
        prepare_output_file(filtered_manifest_path, overwrite=overwrite)
        filtered_fasta_handle = filtered_fasta_path.open('w', encoding='utf-8', newline='\n')
        filtered_manifest_handle = filtered_manifest_path.open('w', newline='', encoding='utf-8')
        filtered_manifest_writer = csv.DictWriter(
            filtered_manifest_handle,
            fieldnames=['identifier', 'full_header', 'sequence_length', 'filter_reason'],
        )
        filtered_manifest_writer.writeheader()

    try:
        with open_text_auto(input_path) as fh:
            for header, seq_lines in fasta_records(fh):
                seq = ''.join(seq_lines)
                seq_len = len(seq)
                total_sequences += 1

                for t in report_thresholds:
                    if seq_len <= t:
                        threshold_counts[t] += 1
                    if seq_len == t:
                        exact_length_counts[t] += 1

                if seq_len < min_length_kept:
                    filtered_sequences += 1
                    if filtered_fasta_handle is not None and filtered_manifest_writer is not None:
                        write_record(filtered_fasta_handle, header, seq_lines)
                        filtered_manifest_writer.writerow(
                            {
                                'identifier': extract_identifier(header),
                                'full_header': header,
                                'sequence_length': seq_len,
                                'filter_reason': f'length < {min_length_kept}',
                            }
                        )
                    continue

                kept_sequences += 1
                min_kept_length = seq_len if min_kept_length is None else min(min_kept_length, seq_len)
                max_kept_length = seq_len if max_kept_length is None else max(max_kept_length, seq_len)
                kept_lengths_seen.add(seq_len)

                if not report_only:
                    bucket_path = bucket_dir / f'len_{seq_len:07d}.fasta'
                    with bucket_path.open('a', encoding='utf-8', newline='\n') as out:
                        write_record(out, header, seq_lines)
    finally:
        if filtered_fasta_handle is not None:
            filtered_fasta_handle.close()
        if filtered_manifest_handle is not None:
            filtered_manifest_handle.close()

    stats = LengthStats(
        input_path=str(input_path.resolve()),
        total_sequences=total_sequences,
        kept_sequences=kept_sequences,
        filtered_sequences=filtered_sequences,
        min_kept_length=min_kept_length,
        max_kept_length=max_kept_length,
        thresholds={str(k): threshold_counts.get(k, 0) for k in sorted(report_thresholds)},
        exact_lengths={str(k): exact_length_counts.get(k, 0) for k in sorted(report_thresholds)},
        filtered_fasta_path=(str(filtered_fasta_path.resolve()) if (filtered_fasta_path and write_filtered_trace and not report_only) else None),
        filtered_manifest_path=(str(filtered_manifest_path.resolve()) if (filtered_manifest_path and write_filtered_trace and not report_only) else None),
    )

    return stats, sorted(kept_lengths_seen)


def build_batches_from_buckets(
    bucket_dir: Path,
    kept_lengths: List[int],
    output_dir: Path,
    output_prefix: str,
    output_suffix: str,
    batch_size: int,
    overwrite: bool,
) -> List[BatchInfo]:
    batch_infos: List[BatchInfo] = []
    batch_index = 0
    current_count = 0
    out_handle: Optional[TextIO] = None
    out_path: Optional[Path] = None
    current_min_length: Optional[int] = None
    current_max_length: Optional[int] = None
    first_header: Optional[str] = None
    last_header: Optional[str] = None

    def close_batch() -> None:
        nonlocal out_handle, out_path, current_count, current_min_length, current_max_length, first_header, last_header, batch_index
        if out_handle is None or out_path is None:
            return
        out_handle.close()
        batch_infos.append(
            BatchInfo(
                batch_index=batch_index,
                file_path=str(out_path.resolve()),
                n_sequences=current_count,
                min_length=current_min_length,
                max_length=current_max_length,
                first_header=first_header,
                last_header=last_header,
            )
        )
        out_handle = None
        out_path = None
        current_count = 0
        current_min_length = None
        current_max_length = None
        first_header = None
        last_header = None

    for seq_len in kept_lengths:
        bucket_path = bucket_dir / f'len_{seq_len:07d}.fasta'
        if not bucket_path.exists():
            continue
        with bucket_path.open('r', encoding='utf-8', errors='replace') as fh:
            for header, seq_lines in fasta_records(fh):
                if out_handle is None:
                    batch_index += 1
                    out_path = output_dir / f'{output_prefix}_batch_{batch_index:05d}{output_suffix}'
                    if out_path.exists() and not overwrite:
                        raise FileExistsError(
                            f"El archivo de salida ya existe: {out_path}. Usa --overwrite si quieres sobrescribir."
                        )
                    out_handle = out_path.open('w', encoding='utf-8', newline='\n')

                write_record(out_handle, header, seq_lines)
                current_count += 1
                current_min_length = seq_len if current_min_length is None else min(current_min_length, seq_len)
                current_max_length = seq_len if current_max_length is None else max(current_max_length, seq_len)
                if first_header is None:
                    first_header = header
                last_header = header

                if current_count >= batch_size:
                    close_batch()

    close_batch()
    return batch_infos


def write_manifest_and_summary(
    output_dir: Path,
    stats: LengthStats,
    batch_infos: List[BatchInfo],
    batch_size: int,
    min_length_kept: int,
    report_only: bool,
) -> None:
    summary_json = output_dir / 'length_sorted_batch_summary.json'
    manifest_csv = output_dir / 'length_sorted_batch_manifest.csv'

    payload = {
        **asdict(stats),
        'batch_size': batch_size,
        'min_length_kept': min_length_kept,
        'report_only': report_only,
        'n_batches_written': len(batch_infos),
        'batches': [asdict(x) for x in batch_infos],
    }

    with summary_json.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    with manifest_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                'batch_index',
                'file_path',
                'n_sequences',
                'min_length',
                'max_length',
                'first_header',
                'last_header',
            ],
        )
        writer.writeheader()
        for info in batch_infos:
            writer.writerow(asdict(info))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            'Filtra proteínas cortas, guarda la traza de las eliminadas, ordena un FASTA de menor a mayor longitud '
            'y construye batches de tamaño configurable (por defecto 500000). También genera un resumen con cuántas '
            'secuencias tienen longitud <= 50, <= 30 y exactamente 50/30 aminoácidos.'
        )
    )
    ap.add_argument(
        '--args-file',
        default=None,
        help=(
            'Ruta a un TXT con argumentos para el script. El archivo puede contener los mismos flags '
            'que pondrías en la terminal, con espacios, saltos de línea, comillas y comentarios #.'
        ),
    )
    ap.add_argument('--input', required=True, help='Ruta al FASTA/.fa/.fasta[.gz] de entrada')
    ap.add_argument('--output-dir', required=True, help='Directorio donde se guardarán los batches y resúmenes')
    ap.add_argument(
        '--batch-size',
        type=int,
        default=500000,
        help='Número de proteínas por batch. Por defecto: 500000',
    )
    ap.add_argument(
        '--min-length-kept',
        type=int,
        default=51,
        help=(
            'Longitud mínima que se conservará. Por defecto 51, que elimina longitudes <= 50. '
            'Si quieres eliminar <= 30, usa 31.'
        ),
    )
    ap.add_argument(
        '--report-threshold',
        type=int,
        action='append',
        default=[30, 50],
        help='Umbral adicional a reportar para contar secuencias <= umbral y exactamente == umbral. Puede repetirse.',
    )
    ap.add_argument(
        '--output-prefix',
        default=None,
        help='Prefijo base opcional para los batches. Si no se indica, usa el nombre del FASTA.',
    )
    ap.add_argument(
        '--report-only',
        action='store_true',
        help='Solo genera el resumen de longitudes. No crea buckets, batches ni archivos de secuencias filtradas.',
    )
    ap.add_argument(
        '--keep-temp',
        action='store_true',
        help='Conserva la carpeta temporal de buckets por longitud.',
    )
    ap.add_argument(
        '--no-write-filtered-trace',
        action='store_true',
        help='No guarda el FASTA y el CSV con las proteínas filtradas por longitud.',
    )
    ap.add_argument(
        '--filtered-fasta-name',
        default='filtered_out_below_min_length.fasta',
        help='Nombre del FASTA donde se guardarán las secuencias eliminadas. Por defecto: filtered_out_below_min_length.fasta',
    )
    ap.add_argument(
        '--filtered-manifest-name',
        default='filtered_out_below_min_length_manifest.csv',
        help='Nombre del CSV con identificador, cabecera completa y longitud de las secuencias eliminadas.',
    )
    ap.add_argument(
        '--overwrite',
        action='store_true',
        help='Sobrescribe archivos de salida si ya existen.',
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    final_argv = expand_args_file(raw_argv)
    args = build_parser().parse_args(final_argv)

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f'No existe el archivo de entrada: {input_path}')
    if args.batch_size <= 0:
        raise SystemExit('--batch-size debe ser > 0')
    if args.min_length_kept <= 0:
        raise SystemExit('--min-length-kept debe ser > 0')

    report_thresholds = sorted(set(args.report_threshold))
    suffix = infer_output_suffix(input_path)
    prefix = args.output_prefix or base_stem(input_path)
    bucket_dir = output_dir / '_tmp_length_buckets'
    write_filtered_trace = (not args.no_write_filtered_trace)
    filtered_fasta_path = output_dir / args.filtered_fasta_name
    filtered_manifest_path = output_dir / args.filtered_manifest_name

    if bucket_dir.exists() and not args.keep_temp and not args.report_only:
        shutil.rmtree(bucket_dir)

    stats, kept_lengths = distribute_by_length(
        input_path=input_path,
        bucket_dir=bucket_dir,
        min_length_kept=args.min_length_kept,
        report_thresholds=report_thresholds,
        report_only=args.report_only,
        write_filtered_trace=write_filtered_trace,
        filtered_fasta_path=filtered_fasta_path,
        filtered_manifest_path=filtered_manifest_path,
        overwrite=args.overwrite,
    )

    batch_infos: List[BatchInfo] = []
    if not args.report_only:
        batch_infos = build_batches_from_buckets(
            bucket_dir=bucket_dir,
            kept_lengths=kept_lengths,
            output_dir=output_dir,
            output_prefix=prefix,
            output_suffix=suffix,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )

    write_manifest_and_summary(
        output_dir=output_dir,
        stats=stats,
        batch_infos=batch_infos,
        batch_size=args.batch_size,
        min_length_kept=args.min_length_kept,
        report_only=args.report_only,
    )

    if bucket_dir.exists() and not args.keep_temp and not args.report_only:
        shutil.rmtree(bucket_dir)

    print(f'Entrada: {input_path}')
    print(f'Total secuencias: {stats.total_sequences}')
    print(f'Conservadas: {stats.kept_sequences}')
    print(f'Filtradas (< {args.min_length_kept} aa): {stats.filtered_sequences}')
    for t in report_thresholds:
        print(f'<= {t} aa: {stats.thresholds.get(str(t), 0)} | == {t} aa: {stats.exact_lengths.get(str(t), 0)}')
    if not args.report_only:
        print(f'Batches escritos: {len(batch_infos)}')
        if write_filtered_trace:
            print(f'Secuencias filtradas FASTA: {filtered_fasta_path.resolve()}')
            print(f'Secuencias filtradas manifest: {filtered_manifest_path.resolve()}')
    print(f'Resumen JSON: {(output_dir / "length_sorted_batch_summary.json").resolve()}')
    print(f'Manifest CSV: {(output_dir / "length_sorted_batch_manifest.csv").resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
