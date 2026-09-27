r"""Build an IRON design into a self-contained full ELF (no xclbin), mirroring
build_design.py but using CompilableDesign.compile(full_elf_path=...) instead
of the xclbin+insts path. Also prints the real "<device>:<sequence>" XRT
kernel name the full-ELF runtime needs (see mlir_aie's xrtruntime hostruntime
_parse_full_elf_kernel_name / _load_full_elf docstrings).

    python build_full_elf.py designs/dense/dx.py out_dir/dx_full.elf [pos]

If given, `pos` overrides mod.SPECIALIZE["pos"] (a dx.py-only CompileTime
param baking the current KV/ptab position into the compiled program, since
the full-ELF ABI has no host-visible instr buffer for attnpos to patch).
"""
from __future__ import annotations
import importlib.util
import sys
import time
from pathlib import Path

import aie.iron as iron
from aie.iron.device import from_name


def main() -> int:
    src = Path(sys.argv[1]).resolve()
    out_elf = Path(sys.argv[2]).resolve()
    out_elf.parent.mkdir(parents=True, exist_ok=True)

    iron.set_current_device(from_name("npu2", n_cols=None))

    spec = importlib.util.spec_from_file_location(src.stem, src)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(src.parent))
    spec.loader.exec_module(mod)

    specialize = dict(mod.SPECIALIZE)
    if len(sys.argv) > 3:
        specialize["pos"] = int(sys.argv[3])
    design = mod.DESIGN.specialize(**specialize)

    t0 = time.time()
    elf_path, _ = design.compile(full_elf_path=str(out_elf))
    dt = time.time() - t0
    print(f"FULL_ELF_OK {src.name} -> {elf_path} ({dt:.1f}s, {Path(elf_path).stat().st_size} B)")
    print(f"kernel_name = {design.compilable._full_elf_kernel_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
