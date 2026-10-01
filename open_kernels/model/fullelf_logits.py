r"""Greedy-token check for harness/dx_fullelf_check --dump: the NPU's final residual per
token -> the reference head (replica_dense.final_logits: final RMSNorm + q4_1 lm_head, fp64)
-> logits, compared with make_decode.py's ref_logits[_t{t}].bin.

    python model/fullelf_logits.py --model-dir DIR --data DATA --dump DUMP --layers 28 0 1 5 20 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from recipes.load import spec_from_model_dir  # noqa: E402
from q4nx import Q4NX  # noqa: E402
import replica_dense as RD  # noqa: E402


def sfx(t):
    return "" if t == 0 else f"_t{t}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True, help="make_decode.py --out")
    ap.add_argument("--dump", required=True, help="dx_fullelf_check --dump")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("positions", type=int, nargs="+")
    a = ap.parse_args()
    md = Path(a.model_dir)
    spec = spec_from_model_dir(md)
    q = Q4NX(md / "model.q4nx")
    q.hidden = spec.hidden
    data, dump = Path(a.data), Path(a.dump)
    bad = 0
    for t in a.positions:
        res = np.fromfile(dump / f"npu_res{a.layers - 1}{sfx(t)}.bin", np.float32)
        ref = np.fromfile(data / f"ref_logits{sfx(t)}.bin", np.float32)[:spec.real_vocab]
        _, lg = RD.final_logits(q, spec, res.astype(np.float64))
        lg = np.asarray(lg)[:spec.real_vocab]
        cos = float(lg @ ref / np.sqrt((lg @ lg) * (ref @ ref)))
        top_n, top_r = np.argsort(-lg)[:5], np.argsort(-ref)[:5]
        ok = top_n[0] == top_r[0]
        bad += not ok
        print(f"pos {t:4d}: logits cos {cos:.7f}  argmax {top_n[0]} vs ref {top_r[0]} {'OK' if ok else 'MISMATCH'}"
              f"  top5 {top_n.tolist()} / ref {top_r.tolist()}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
