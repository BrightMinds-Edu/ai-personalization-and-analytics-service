#!/usr/bin/env python3
"""
convert_to_contents.py
──────────────────────
Read either a CSV (Question / Explanation) *or* an old `.jsonl` that still uses
"messages":[…], and emit a new `.jsonl` whose top-level key is **"contents"**
with the {role, parts:[{text:…}]} structure.

USAGE
-----
# ➊  From a CSV you just created
python convert_to_contents.py math.csv math_contents.jsonl

# ➋  From an existing jsonl that has "messages"
python convert_to_contents.py old_messages.jsonl new_contents.jsonl
"""

import csv, json, pathlib, sys, itertools
from typing import Iterable, Dict, Any

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def mk_contents(question: str, answer: str) -> dict:
    """Return one record that fits the 'contents' schema."""
    return {
        "contents": [
            {
                "role": "user",
                "parts": [ { "text": question } ]
            },
            {
                "role": "model",
                "parts": [ { "text": answer } ]
            }
        ]
    }


def convert_csv(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """Yield objects from a CSV that has Question, Explanation columns."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if {"Question", "Explanation"} - set(reader.fieldnames or []):
            raise ValueError("CSV needs columns named 'Question' and 'Explanation'")
        for row in reader:
            q = row["Question"].strip()
            a = row["Explanation"].strip()
            yield mk_contents(q, a)


def convert_messages_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """Yield objects converted from an old messages-style jsonl."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Expect messages[0].content  / messages[1].content
            msgs = obj.get("messages")
            if not msgs or len(msgs) < 2:
                raise ValueError("Line missing 'messages' array with 2 items.")
            q = msgs[0]["content"] if "content" in msgs[0] else msgs[0]["parts"][0]["text"]
            a = msgs[1]["content"] if "content" in msgs[1] else msgs[1]["parts"][0]["text"]
            yield mk_contents(q, a)


def main(src: str, dest: str):
    src_path = pathlib.Path(src)
    dest_path = pathlib.Path(dest)

    # Choose converter based on extension / first line
    if src_path.suffix.lower() == ".csv":
        gen = convert_csv(src_path)
    else:
        # Peek first line to decide
        first = src_path.open(encoding="utf-8").readline()
        gen = convert_messages_jsonl(src_path) if '"messages"' in first else None
        if gen is None:
            raise ValueError("Unsupported source format.")

    with dest_path.open("w", encoding="utf-8") as out:
        for obj in gen:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅  Wrote {dest_path} ({sum(1 for _ in dest_path.open()):,} lines)")


if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        sys.exit("Usage: convert_to_contents.py input.[csv|jsonl] [output.jsonl]")
    src = sys.argv[1]
    dest = sys.argv[2] if len(sys.argv) == 3 else pathlib.Path(src).with_suffix(".contents.jsonl")
    main(src, dest)



# #!/usr/bin/env python3
# """
# csv2jsonl.py  –  Convert a two-column CSV (Question, Explanation) into JSON-Lines.

# Usage
# -----
# $ python csv2jsonl.py                               # uses the hard-coded default paths
# $ python csv2jsonl.py path/to/input.csv             # auto-names output .jsonl
# $ python csv2jsonl.py in.csv out.jsonl              # explicit I/O

# Expected CSV headers
# --------------------
# Question , Explanation
# """

# import csv
# import json
# import pathlib
# import sys
# from typing import Sequence


# # ──────────────────────────────────────────────────────────────────────
# # Helpers
# # ──────────────────────────────────────────────────────────────────────
# REQUIRED_COLS: set[str] = {"Question", "Explanation"}


# def row_to_jobj(question: str, explanation: str) -> dict:
#     """Return one JSON object in the desired chat-messages format."""
#     return {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": f"TRANSCRIPT: {question}\n\n LABEL:"
#             },
#             {
#                 "role": "model",
#                 "content": explanation
#             },
#         ]
#     }


# def convert(csv_path: pathlib.Path, jsonl_path: pathlib.Path) -> None:
#     """Read the CSV and write JSON-Lines to *jsonl_path*."""
#     with csv_path.open(newline="", encoding="utf-8") as inf, \
#          jsonl_path.open("w", encoding="utf-8") as outf:

#         reader = csv.DictReader(inf)
#         missing = REQUIRED_COLS - set(reader.fieldnames or [])
#         if missing:
#             sys.exit(f"❌  CSV is missing columns: {', '.join(missing)}")

#         for row in reader:
#             jobj = row_to_jobj(
#                 row["Question"].strip(),
#                 row["Explanation"].strip()
#             )
#             outf.write(json.dumps(jobj, ensure_ascii=False) + "\n")

#     print(f"✅  Wrote {jsonl_path} ({jsonl_path.stat().st_size:,} bytes)")


# # ──────────────────────────────────────────────────────────────────────
# # Main
# # ──────────────────────────────────────────────────────────────────────
# def main(argv: Sequence[str] | None = None) -> None:
#     argv = list(argv or sys.argv[1:])

#     # Defaults – edit to taste
#     default_csv = pathlib.Path("csv_files/2021 - sinhala gram fine tune.csv")
#     default_jsonl = default_csv.with_suffix(".jsonl")

#     if not argv:
#         in_path, out_path = default_csv, default_jsonl
#     elif len(argv) == 1:
#         in_path = pathlib.Path(argv[0])
#         out_path = in_path.with_suffix(".jsonl")
#     elif len(argv) == 2:
#         in_path, out_path = map(pathlib.Path, argv)
#     else:
#         sys.exit("Usage: csv2jsonl.py [input.csv] [output.jsonl]")

#     convert(in_path, out_path)


# if __name__ == "__main__":
#     main()
