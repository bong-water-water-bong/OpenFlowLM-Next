r"""Build an IRON design into a self-contained full ELF (no xclbin), mirroring
build_design.py but using CompilableDesign.compile(full_elf_path=...) instead
of the xclbin+insts path. Also prints the real "<device>:<sequence>" XRT
kernel name the full-ELF runtime needs (see mlir_aie's xrtruntime hostruntime
_parse_full_elf_kernel_name / _load_full_elf docstrings).

    python build_full_elf.py designs/dense/dx.py out_dir/dx_full.elf [pos]
    python build_full_elf.py designs/dense/dx.py out_dir/dx_rt.elf --rtpos

If given, `pos` overrides mod.SPECIALIZE["pos"] (a dx.py-only CompileTime
param baking the current KV/ptab position into the compiled program, since
the full-ELF ABI has no host-visible instr buffer for attnpos to patch).

--rtpos builds dx.py's run-time-position variant instead: ONE ELF for every
position. The host writes three words of the run's control scratchpad per
dispatch (dx.py RTPOS_PARAMS). Two of them are DMA offsets that IRON's
offset_parameter lowers by itself. The third, the KV window's row count, sets
a transfer LENGTH, which neither IRON nor aiecc can take from the scratchpad,
so this script adds it: it stops aiecc after the runtime sequence's DMAs are
lowered to NPU register writes (--checkpoint / --cut npu_dma_lowered.mlir),
inserts one `aiex.npu.update_from_scratchpad<mul>` right after the window
BD's words are written -- adding dx_win_extra * KV_ROW/4 to the BD's
buffer-length word (word 0 of a shim BD, counted in 32-bit words) -- and
resumes aiecc from the edited checkpoint. update_from_scratchpad is the
firmware's UPDATE_REG op, the one offset_parameter itself uses on the BD's
address word. It always rewrites a register PAIR; on (length, address-low)
it writes the address word back with its own low 16 bits (the sum cannot
carry out of the length word), so only the length moves. Next to the ELF it
writes <elf>.params.txt: "<n>" then "<name> <state_table_idx> <type> <kind>"
per parameter (the format of mlir-aie's test_utils::ParameterScratchpad;
all three are written raw, no << 2), and <elf>.rtpos: "kv_row B ptab_row B
rows N" (the byte multipliers for the two offsets, and the cache's rows).
"""
from __future__ import annotations
import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path

import aie.iron as iron
from aie.iron.device import from_name
from aie.utils import config
from aie.utils.compile.utils import compile_mlir_module

CUT = "npu_dma_lowered.mlir"


def _param_indices(mlir: str) -> dict[str, tuple[int, str, str]]:
    """name -> (state_table_idx, type, kind) from the lowered module's declarations."""
    out = {}
    for line in mlir.splitlines():
        m = re.search(r'aiex\.scratchpad_parameter @(\w+)\b', line)
        if not m:
            continue
        idx = re.search(r'state_table_idx = (\d+)', line)
        ty = re.search(r':\s*(i\d+|bf16)\b', line)
        kind = re.search(r'kind = (\d+)', line)            # ScratchpadParameterKind: 0 core, 1 addr
        if not idx:
            raise RuntimeError(f"rtpos: no state_table_idx on: {line.strip()}")
        out[m.group(1)] = (int(idx.group(1)), ty.group(1) if ty else "i32", ("addr" if kind and kind.group(1) == "1" else "core"))
    return out


def _patch_window_length(mlir: str, idx: int, kv_arg: int, kv_row: int, column: int) -> str:
    """Insert the window-length update after the window BD's address patch.

    The window BD is the one shim BD that reads the KV buffer (`kv_arg`) on the
    attention input column: its words go out as one 8-word blockwrite at the BD's
    base address, then the address_patch at base + 4."""
    pat = re.compile(r'^(\s*)aiex\.npu\.address_patch\([^)]*\) \{addr = (\d+) : ui32, arg_idx = '
                     + str(kv_arg) + r' : i32\}[^\n]*$', re.M)
    hits = [h for h in pat.finditer(mlir) if (int(h.group(2)) >> 25) == column]
    if len(hits) != 1:
        raise RuntimeError(f"rtpos: expected one KV-buffer BD on column {column}, found {len(hits)}")
    h = hits[0]
    base = int(h.group(2)) - 4                                   # the BD's word 0 (buffer length)
    # The blockwrite right before it writes that BD, with a one-row length.
    bw = list(re.finditer(r'%(\S+) = memref\.get_global @(\S+) : memref<8xi32>[^\n]*\n'
                          r'\s*aiex\.npu\.blockwrite\(%\1\) \{address = (\d+) : ui32\}', mlir[:h.start()]))
    if not bw or int(bw[-1].group(3)) != base:
        raise RuntimeError(f"rtpos: the window BD's blockwrite (address {base}) does not precede its patch")
    g = re.search(r'memref\.global "private" constant @' + re.escape(bw[-1].group(2))
                  + r'\s*:\s*memref<8xi32> = dense<\[([^\]]*)\]>', mlir)
    if not g or int(g.group(1).split(",")[0]) != kv_row // 4:
        raise RuntimeError(f"rtpos: window BD word 0 is not one row ({kv_row // 4} words): "
                           f"{g.group(1) if g else 'no global'}")
    ins = (f"\n{h.group(1)}aiex.npu.update_from_scratchpad<mul> {{address = {base} : ui32, "
           f"func_arg = {kv_row // 4} : ui32, state_table_idx = {idx} : ui8}}")
    return mlir[:h.end()] + ins + mlir[h.end():]


def build_rtpos(mod, design, out_elf: Path) -> None:
    design.compile(full_elf_path=str(out_elf))      # the design's own build: objects, aie.mlir
    prj = out_elf.parent / f"{out_elf.stem}.prj"
    try:
        ck = prj / "rtpos_ck"
        flags = list(design.compilable.aiecc_flags) + [f"--checkpoint={ck}", f"--cut={CUT}"]
        compile_mlir_module((prj / "aie.mlir").read_text(), full_elf_path=out_elf, work_dir=prj, options=flags)
        f = ck / CUT / CUT
        mlir = f.read_text()
        params = _param_indices(mlir)
        missing = [n for n in mod.RTPOS_PARAMS if n not in params]
        if missing:
            raise RuntimeError(f"rtpos: parameters not declared in {CUT}: {missing}")
        idx, ty, _ = params[mod.RTPOS_PARAMS[2]]
        # The scratchpad pass calls it "core" (no DMA offset names it), but it feeds a BD
        # register like an "addr" one, so the host writes it raw (no << 2).
        params[mod.RTPOS_PARAMS[2]] = (idx, ty, "addr")
        ain_col = 2                                      # dx.py: of_ain.prod(tile=Tile(2, 0))
        f.write_text(_patch_window_length(mlir, idx, 3, mod.L.KV_ROW, ain_col))
        r = subprocess.run([config.aiecc_path(), f"--resume={ck / 'manifest.json'}", "--no-progress"],
                           cwd=prj, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"aiecc --resume failed:\n{r.stdout}\n{r.stderr}")
        rows = sorted((i, n, t, k) for n, (i, t, k) in params.items())
        (out_elf.parent / f"{out_elf.name}.params.txt").write_text(
            f"{len(rows)}\n" + "".join(f"{n} {i} {t} {k}\n" for i, n, t, k in rows))
        L = mod.L                                        # the row geometry a host needs for the values
        nrows = min(L.KV_BYTES // L.KV_ROW, L.PTAB_BYTES // L.PTAB_ROW)
        (out_elf.parent / f"{out_elf.name}.rtpos").write_text(
            f"kv_row {L.KV_ROW} ptab_row {L.PTAB_ROW} rows {nrows}\n")
        print(f"rtpos: window length += {mod.RTPOS_PARAMS[2]}[{idx}] * {mod.L.KV_ROW // 4} words; "
              f"params {rows}")
    except Exception:
        out_elf.unlink(missing_ok=True)                  # never leave the unpatched (1-row window) ELF behind
        raise


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--rtpos"]
    rtpos = len(args) != len(sys.argv) - 1
    src = Path(args[0]).resolve()
    out_elf = Path(args[1]).resolve()
    out_elf.parent.mkdir(parents=True, exist_ok=True)

    iron.set_current_device(from_name("npu2", n_cols=None))

    spec = importlib.util.spec_from_file_location(src.stem, src)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(src.parent))
    spec.loader.exec_module(mod)

    specialize = dict(mod.SPECIALIZE)
    if len(args) > 2:
        specialize["pos"] = int(args[2])
    if rtpos:
        specialize["rtpos"] = 1
    design = mod.DESIGN.specialize(**specialize)

    t0 = time.time()
    if rtpos:
        build_rtpos(mod, design, out_elf)
        elf_path = out_elf
    else:
        elf_path, _ = design.compile(full_elf_path=str(out_elf))
    dt = time.time() - t0
    print(f"FULL_ELF_OK {src.name} -> {elf_path} ({dt:.1f}s, {Path(elf_path).stat().st_size} B)")
    print(f"kernel_name = {design.compilable._full_elf_kernel_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
