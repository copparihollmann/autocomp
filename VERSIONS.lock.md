# VERSIONS.lock — the coherent stack for whole-Radiance (Muon SIMT + MX-Gemmini) work

**Baseline: the taped-out silicon.** RTL is pinned to radiance `tapeout-330`
(<https://github.com/ucb-bar/radiance/releases/tag/tapeout-330>). Everything else is pinned to
be *coherent with that silicon* — see `COHERENCE.md` for the invariant and how it is checked.

Last updated: 2026-07-13. Backup of the pre-change state: `/scratch2/agustin/mx-backup-tapeout330`.

## The tuple

| Component | Path | Commit | Branch / tag | Role |
|---|---|---|---|---|
| **radiance (RTL)** | `chipyard/generators/radiance` | `2a332ab` | **`tapeout-330`** (branch `tapeout-330-mx`) | The taped-out RTL. THE reference. |
| **gemmini (RTL)** | `chipyard/generators/gemmini` | `69a1c03` | **`tapeout-260329`** (`origin/gemmini-mx`) | The taped-out MX-Gemmini RTL. |
| **cyclotron** | `radiance/cyclotron` | `404db1b` | `feat/radiance-autocomp` | Fast sim + MX co-model, AND the RTL's DPI tracer. |
| **chipyard** | `chipyard` | `18d993d0` | `feat/radiance-autocomp` | Superproject / build. |
| **radiance-kernels** | `radiance-kernels` | `0b325ee` | `feat/radiance-autocomp` | Kernels + the mxgemm driver. |
| **mxgemmini (kernel lib)** | `radiance-kernels/lib/mxgemmini` | `62c4f85` | `6fc8ec7-radiance-tapeout` | MX kernels + golden models. |
| **autocomp** | `autocomp` | `0236b7d` | `feat/radiance-autocomp` | Harnesses, goldens, eval, agent. |

All branches are pushed to `github.com/copparihollmann/<repo>`.

## Pinned parameters (do not drift)

| Param | Value | Why |
|---|---|---|
| `DIM` | 16 | 16×16 systolic mesh. |
| `BANK_NUM × BANK_ROWS` | 4 × **2048** | Tapeout SMEM is **128 KiB** (`TapeoutSmemConfig`: `size=128<<10`, `numBanks=4`). |
| MX co-model | `CYCLOTRON_MXGEMMINI=1` | Off by default; inert in the RTL DPI path. |
| Calibrated `KCMP` / `KDMA` | ~960 / ~76 | ⚠️ Fitted against the OLD (+35) simv. **Must be re-fitted against the tapeout-330 simv.** |

## Decisions worth remembering (each one bit us)

1. **mxgemmini is pinned to `6fc8ec7` (+ an `abs()` fix ⇒ `62c4f85`), NOT the dev tip `d3b3d10`.**
   `d3b3d10` sets `BANK_ROWS=4096` for a *256 KiB rocket/spike* config. The tapeout is 128 KiB.
   `6fc8ec7` already has `BANK_ROWS=2048` **and** already contains the new 128×128×512 kernels and
   the fully-matched goldens, so nothing is lost.
   A `-DBANK_ROWS=2048` compile override **cannot** fix a bad pin: `gemmini_params.h` `#define`s it
   unconditionally, so the header always wins. **The submodule pin is the only lever.**
   Guard: `static_assert(BANK_NUM*BANK_ROWS*DIM == 128 KiB)` in `mxgemm_lib.hpp`.

2. **The RTL's DPI tracer is OUR cyclotron (`404db1b`), not tapeout-330's pin (`5eac109`).**
   `5eac109` has **no `trace_db.rs`** — using it would break `rtl_kernel_cycles.py` and
   `kernel_utilization.py` (all cycle/utilization measurement). It is safe to use ours:
   `CyclotronTile.scala` (the DPI interface) is *identical* between tapeout-330 and our HEAD, and
   RTL fidelity comes from the **Verilog**, not from the co-simulator. The simv links
   `libcyclotron.so` dynamically (rpath → `cyclotron/target/debug`), so it is a *runtime* dep.

3. **The RTL simv must be elaborated from tapeout-330.** The previous simv (preserved as
   `sims/vcs/*.plus35.bak`) was built 2026-06-03 from radiance `293aed3` = tapeout-330 **+35**, so
   *no earlier RTL result was actually tapeout-faithful*.

4. **`SPAD_DEST` is computed, never hardcoded.** See `COHERENCE.md`; the old `= 256` silently
   corrupted every 128×128/256×256 kernel.

## Rebuilding from this lock

```bash
# RTL (tapeout-faithful)
cd chipyard && source env.sh                      # JDK-20; system JDK-21 breaks sbt
git -C generators/radiance checkout tapeout-330-mx
make -C sims/vcs CONFIG=RadianceSingleClusterConfig

# fast sim + co-model
cd generators/radiance/cyclotron && cargo build --release

# kernels
cd radiance-kernels && git submodule update --init lib/mxgemmini
```

Golden models need `torch` + `qtorch` (installed in `autocomp/.venv`).
