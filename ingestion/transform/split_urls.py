#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def dump_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Separa URLs internas y externas a partir de un archivo JSONL.")
    parser.add_argument("--input", required=True, help="Archivo JSONL de entrada")
    parser.add_argument("--internal-domain", default="infozone.atlassian.net", help="Dominio interno principal")
    parser.add_argument("--output-dir", default="data/discovery", help="Directorio de salida")
    parser.add_argument("--internal-name", default="urls_internal.jsonl")
    parser.add_argument("--external-name", default="urls_external.jsonl")
    args = parser.parse_args()

    rows = list(load_jsonl(Path(args.input)))
    internal = []
    external = []

    for row in rows:
        url = (row or {}).get("url", "")
        host = urlparse(url).netloc.lower()
        if not host:
            continue
        if host == args.internal_domain.lower():
            internal.append(row)
        else:
            external.append(row)

    out_dir = Path(args.output_dir)
    dump_jsonl(out_dir / args.internal_name, internal)
    dump_jsonl(out_dir / args.external_name, external)

    print(f"Input rows: {len(rows)}")
    print(f"Internal rows: {len(internal)} -> {out_dir / args.internal_name}")
    print(f"External rows: {len(external)} -> {out_dir / args.external_name}")


if __name__ == "__main__":
    main()
