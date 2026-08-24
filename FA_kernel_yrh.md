# MXFP8 flash-attention KERNEL bring-up (Muon + mxGemmini)

## ✅ RESOLVED — full kernel validated end-to-end on RTL (32x32x32)
O matches the quantized golden (golden_O_normfirst) at **2.48% Frobenius** rel err
(6.2% vs the true fp32 reference golden_O_u16) — reasonable for two cascaded MX-FP8
mesh passes + an fp8 requant (QK alone is 0.89%). Signs all correct, max abs diff 0.039.

WINNING ARCHITECTURE (all-SMEM, fence.s-coherent, NO GMEM round-trip, NO global fence):
1. QK^T: `mxgemm_single_output_tile<QK>` -> S bf16 in SMEM @ SPAD_DEST (0x1000).
2. SIMT online softmax (reads S from SMEM) -> normalized P (=exp(S-m)/l) bf16 -> P_SMEM.
3. SIMT MX-FP8 requant `requant_P_to_spad_tiled` (PARALLEL over rows):
   - P e4m3 elements -> A scratchpad (spad base 0) in the Gemmini TILED layout:
     byte_off(m,c) = (m/16 * (SK/16) + c/16)*256 + (m%16)*16 + (c%16); written 4-e4m3/
     word (each row = two disjoint word-aligned 16-byte runs -> no cross-thread overlap).
   - per-row E8M0 scale (floor_log2(blockmax), emax_target=0) -> word-per-scale SMEM
     scratch (SCALE_SMEM 0xD000), then `pack_scales_to_sfmem` packs them (4/word, thread-0)
     into the A scale SRAM GEMMINI_SF_MEM_A (0x8A000), linear byte i = A row i.
   - e4m3 conversion uses TRUNCATION (`bf16_to_e4m3<RNE=false>`) to match the golden's
     float_quantize_trunc (RNE gave 6.1%, trunc gives 2.48%).
4. PV: `mxgemm_single_output_tile<PV, false, SKIP_A=true>` — skips BOTH the A move-in
   (A=P already in spad 0) AND the A scale-load (scales already in SF-SRAM); B=V + B_scales
   from GMEM DMA (static loader data, always visible). Result O bf16 -> SPAD_DEST.
5. SIMT `copy_smem_to_gmem_simt` move-out S_SMEM(=SPAD_DEST) -> O_GMEM 0x40040000.

KEY LESSONS (corrects earlier hypotheses in the history below):
- The earlier "PV SKIP_A hangs" was WRONG: both runs were timing out in a catastrophically
  slow THREAD-0-SERIAL requant (1024 heavy bf16 convs). PARALLELIZING requant over rows
  (each thread owns whole rows, uniform control flow) fixed it -> kernel completes ~170-330K cyc.
- SIMT-written GMEM is NOT reliably visible to the Gemmini DMA: fence.s is SMEM-only;
  plain `mu_fence` is unreliable (SFUPipe.scala:71 assert(!reqSent) when warps issue it
  unsynchronized; can also spin). So the robust path keeps P ELEMENTS in spad and P SCALES
  in SF-SRAM (both SMEM, fence.s-coherent) and uses SKIP_A — NO GMEM, NO mu_fence.
- Vectorized SUB-WORD (byte) stores from multiple lanes to adjacent addresses compile to
  OVERLAPPING word stores that clobber neighbours (this corrupted the scale array -> O*2^-127).
  Fix: write scales word-per-slot to SMEM scratch, then thread-0 packs 4/word (non-overlapping).
- A-spad byte layout + linear SF-SRAM A-scale layout confirmed from RTL/DMA move-in code.


Kernel dir: `radiance-kernels/kernels/flash_attention_mx/`. Builds on the verified
golden model (lib/mxgemmini/flash_attention_model.py) and the working gemm_mxgemmini
path. Incremental: QK^T gemm (done) -> SIMT softmax+requant -> PV gemm.

## Files
- `fa_gen_data.py`     -> include/fa_data.h : Q/K/V fp8 + E8M0 scales in mxgemm layout
  (QK^T: A=Q[Sq,d], B=K^T[d,Sk], A_scales_row[d/32][Sq], B_scales_col[d/32][Sk]) + V_in,
  V_scales + golden QK_S_bf16/O. Reuses fp8_matmul_model encoders. Sizes parametric.
- `fa_gen_goldens.py`  -> golden_*.npy (S, scaled-S, m, l, P fp8, P scales, O) per stage.
- `mxgemm_core.hpp`    : pointer-parameterized copy of mxgemm_lib.hpp. mxgemm()/
  mxgemm_single_output_tile()/copy_gmem_to_smem_async() now take A_in/B_in/A_scales/
  B_scales base pointers (the original lib hardcodes global symbols; 27 call-sites, so
  kept separate). Lets QK^T and PV both drive the same gemm core with different bases.
- `flash_attention_mx.cpp` : kernel. FA-K3 = single QK^T gemm -> bf16 S at 0x40000000.
- `fa_verify_out.py`   : parses the cyclotron .out instruction trace, reconstructs a
  GMEM tensor from the SIMT move-out store trace, compares to a golden .npy.
- `Makefile`           : MU_CFLAGS += -include gemmini_host_shim.h (printf/exit/abs).
- `host.cpp`           : copied from gemm_mxgemmini (polls GPU finished).

## Key facts learned
- mxgemm consumes O(1) e4m3 elements; my mx_quantize_cols (golden) already does this.
- Data arrays are static-const in kernel .rodata; placed at device addrs (e.g.
  QK_A_in@0x10005000). Gemmini DMA ORs RAD_HOST_GPU_DRAM_BASE (0x1_00000000).
- **Muon SIMT global store** (the move-out): opcode 0x23, but it's a 32-bit *vectorized*
  store where each lane's effective address = rs1.data[lane] directly (NO S-type imm),
  data = rs2.data[lane] (a 32-bit word = 2 packed bf16). fa_verify_out.py relies on this.
- Verify offline from the .out trace, NOT an in-kernel read-back: 256 serial global
  loads on one thread cost ~500k cycles in debug sim (each load is a full-hierarchy
  round-trip). Keep kernels = compute+move-out; verify externally.
- Run with 2 warps (mu_schedule(..,2)) -> low occupancy, sidesteps the gemm_simt L0d
  TLNBDCache issue entirely (Gemmini does the matmul; only 2 warps do SIMT move-out).
- Build/run gotcha: `run-binary` (non-debug) tried to REBUILD the non-debug simv and
  failed because kernel-build-env.sh's CPLUS_INCLUDE_PATH (muon cxx-v1 headers) leaks
  into the host Cyclotron g++ build (chrono/atomic errors). Use the prebuilt **debug**
  simv (run-binary-debug) which doesn't rebuild. (TODO: scope CPLUS_INCLUDE_PATH to the
  muon build only so non-debug simv can rebuild cleanly.)

## FA-K3 status: QK^T VALIDATED ON RTL
S = Q@K^T via parameterized mxgemm runs to clean **$finish** (no assert) on
RadianceTapeoutSimConfig (123.8 us, 61k cycles, 2 warps). Full S tensor reconstructed
from the move-out store trace (fa_verify_out.py): **0.89% Frobenius rel err vs the
golden model** (mean abs 0.036 on +/-31 range; 49% exact bf16-code match, 62% within
1 ULP) -- the residual is RTL-MAC-ordering vs python-model rounding. QK^T MX FP8
computes correctly. Config: FP8, TILE_M=TILE_N=128, TILE_K=d=64 (single K-tile,
2 scale groups), QUANT_OUTPUT=false (bf16 out).
- Verifier note: SIMT global store is opcode 0x23, ea = rs1.data[lane] directly, data
  = rs2.data[lane]; the trace tmask field must NOT be used to gate lanes (mis-parses).
- bf16->e4m3 requant encoder prototyped + validated vs golden (~100% on bf16 inputs);
  ported to flash_mx_impl.hpp::bf16_to_e4m3 for the softmax requant (FA-K4).

## FA-K4 status: softmax+requant WRITTEN, compiles, but HANGS in sim (open)
flash_mx_impl.hpp::softmax_requant integrated after the QK^T gemm (reads S from
GMEM 0x40000000, writes P fp8 -> 0x40010000, P scales -> 0x40020000, l fp32 ->
0x40030000). bf16_to_e4m3 encoder validated vs golden (~100% on bf16). Reductions
use the proven softmax-kernel tree pattern (no in-loop fences) after an initial
muon-backend crash (UnifyLoopExits "Unsupported block terminator") from in-loop
fences / data-dependent branches -- FIXED by tree reduction + block=i/2 strided-col
indexing (no data-dependent branch).
RUN RESULT: kernel builds; on VCS the QK^T phase runs but the **softmax phase STALLS**
(GPU stops issuing mid-softmax around pc 0x100039xx; CPU spins polling GPU-finished;
no assert, no $finish; 0 P/l stores produced even at 32x32x32). softmax_requant has
no mu_barrier, so it's a **memory-system stall** -- a global load of S (or store of P)
that never completes. Hypotheses to chase with the FSDB/rtl-debug skill:
  1. softmax's multiple outstanding global LOADS of S strand a response in the L0d
     TLNBDCache (same makeLandingPads=false root cause as gemm_simt, but manifesting
     as a hang rather than the sim-only assert) -- try: serialize loads / 1 outstanding,
     or read S from SMEM (the gemm C tile) instead of GMEM round-trip.
  2. SMEM scratch at 0xC000 or the P byte-store path stalls the LSU.
UPDATE: switching S reads from 16-bit (sub-word) to 32-bit word loads did NOT fix the
hang (still spins, 0 P/l stores at 32x32x32). So it is not sub-word loads -- most
likely an L0d response stranded on the softmax's outstanding global LOADS of S (same
makeLandingPads=false fragility as gemm_simt, here a hang not the sim assert).
Recommended next: run debug, FSDB-trace core.be.io_dmem_req_*/resp_* at the stall time
(rtl-debug skill) to confirm which load is outstanding; then redesign to read S from
SMEM (the gemm C result tile, no GMEM round-trip) or bound to 1 outstanding load.
NEXT STEP: FSDB-trace the core dmem req/resp at the stall time (rtl-debug skill) to
see which load/store is outstanding; then redesign the softmax memory access (most
likely: keep S in SMEM, avoid the GMEM round-trip; or bound outstanding loads).
Verifier ready: fa_verify_out.py (S bf16, P u8 via --elem-bytes 1) + fa_verify_l.py (l).

## FA-K4/K5 UPDATE (SMEM-resident redesign, per user directive)
Architecture per user: gemm bf16 output stays in SMEM (SPAD_DEST), SIMT activates,
then re-quantize via the HW requantizer SMEM region (GEMMINI_REQUANT=0x40000) -> fp8
at SMEM 0x0 + E8M0 scales. Progress:
- **Hang root-caused + FIXED**: softmax reading S from GMEM stalled -> now reads S from
  SMEM SPAD_DEST (0x1000, the QK^T result, row-major bf16). ALSO: plain `fence`
  (mu_fence) STALLS on muon (only fence.s/gemmini_fence are used by working kernels) --
  removed it. With both fixes the full pipeline (QK gemm -> SIMT softmax -> PV gemm ->
  SIMT normalize) runs to clean $finish.
- **Validated so far**: QK^T S 0.89% rel err; softmax row-denominator l 0.56% rel err
  (softmax math correct: scale=1/sqrt(d) emitted as bf16 by the generator).
- **OPEN**: end-to-end O is wrong (~98% err, near-zero) because P is round-tripped
  through GMEM (SIMT byte/half stores) and the PV gemm's Gemmini DMA reads it before
  the SIMT stores are ordered -- mu_fence_smem only orders SMEM, and plain mu_fence
  stalls. => MUST keep P in SMEM and feed PV without a GMEM round-trip, i.e. use the
  HW requantizer: softmax writes P bf16 (packed words) to SMEM scratch -> single warp
  copies to GEMMINI_REQUANT -> HW emits P fp8 at SMEM 0x0 (=A scratchpad) + scales ->
  PV gemm skips the A DMA (A already in spad). See requant.cpp for the exact pattern.
- Verifiers: fa_verify_l.py (l, fp32, tmask=char-per-lane), fa_verify_out.py (S/O bf16
  word stores). NOTE: tmask in the .out trace is one CHAR per lane ("1111..1"=all 16),
  not a bitmask. P/scale byte stores merge to storeHalf -> hard to parse; verify via O.

## RTL-CONFIRMED root cause (read MxRequantizer.scala + ScaleFactorMem.scala)
Definitive scale dataflow (file:line in mission-control thread):
- MxRequantizer.scala:563-567: the requantizer writes per-block E8M0 scales ONLY via DMA
  to GMEM at the gemmini_mxquant_config_mvout base addr (= my C_scale_factors), gated by
  `scale_buffer_full`. NOT into the SF scale SRAM.
- ScaleFactorMem.scala:189-241: during a matmul the mesh reads per-block scales FROM the
  SF scale SRAM (GEMMINI_SF_MEM_A 0x8A000 / _B 0x88000), banks 4-7 = A/act, 0-3 = B/wt.
- => chaining requires: requant scales -> GMEM -> load_scale_factors() -> SF SRAM ->
  next gemm. There is NO fused path and (per the RTL analysis) no example kernel that
  re-feeds a requantized tile into a subsequent matmul; requant.cpp only copies fp8 out.
- OBSERVED: C_scale_factors stays ZERO after writing P to GEMMINI_REQUANT, and keeping
  the SF SRAM (skip A-scale load) gives NaN. So the standalone GEMMINI_REQUANT write
  produces the fp8 ELEMENTS (spad 0) but does NOT produce/emit the per-block SCALES --
  the scale path appears tied to a gemm move-out (QUANT_OUTPUT) requant, not the bare
  requant-region write. So P (softmax output, not a gemm result) has no path to get HW
  per-block scales via the standalone requantizer.
CONCLUSION: the user-specified "write P to the requant SMEM region" gives correct fp8
but NOT the chaining scales on this silicon; HW-requant->mesh chaining for an arbitrary
SIMT-produced tile is unsupported/undocumented. Two viable completions (both substantial,
next session):
  (A) SIMT e4m3 requant (my validated encoder: correct O(1) fp8 + E8M0 scales) and place
      P fp8 in the PV A-spad (tiled) + P scales directly into SF SRAM A (0x8A000) in the
      ScaleFactorMem layout (read_row = loop_bound_i*(k>>1)+i), then PV with SKIP_A and
      skip-A-scale-load. All SIMT->SMEM, ordered by fence.s (works). Needs the exact
      A-spad tiling + SF-SRAM scale layout (from ScaleFactorMem.scala).
  (B) Requantize P via a QUANT_OUTPUT gemm move-out (so the scale DMA fires) -- but P is
      not a gemm result, so this needs an identity/staging gemm.
Path (A) is the most promising and uses only validated pieces + known SMEM-ordering.

## (history) requant->PV scale-chaining attempts
Tried both scale-handling variants for PV consuming the requant output:
- load A_scales from C_scale_factors (= all zero, requant didn't DMA there) -> O = 0x7e40
  (huge finite).
- SKIP_A also skips the A-scale load (keep whatever the requant left in the SF scale
  SRAM) -> O = 0x7fc0 (NaN).
Both garbage -> the requant's per-block scales are NOT being applied to the PV mesh
correctly by any approach I can construct, and/or the requant's elements overflow the
mesh. The MxRequantizer chaining protocol (where it emits scales, how the next gemm's
mesh consumes them, whether SKIP_A must also skip the A-scale load AND where to point
A_scales) is genuinely unexercised by any reference (requant.cpp only copies fp8 out)
and is the blocking unknown. This is an RTL-owner question, not further guessable in SW.
DONE this session: QK^T 0.89%, softmax P 0.74% (RTL-validated); O-extraction SIMT
corruption SOLVED (GMEM requant feed); full QK->softmax->requant->PV->O pipeline runs to
$finish with O landing in GMEM. Remaining: the requant->PV scale wiring (needs protocol).

## (history) requant scales never reach C_scale_factors
Dumped C_scale_factors (the GMEM addr configure_mxgemmini<RQ> gave to
gemmini_mxquant_config_mvout) after the requant feed: it is ALL ZERO. So the
requantizer's E8M0 scale DMA never populated it. PV then reads zero/garbage A-scales
-> the mesh applies a bad scale -> O = 0x7e40 (huge). (So the earlier "element
overflow" framing was wrong; the elements are fine, the SCALES are missing.)
So the real open question is the MxRequantizer SCALE-OUTPUT protocol for a CHAINED
consumer: writing P bf16 -> GEMMINI_REQUANT produces the fp8 output (spad 0) but does
NOT seem to DMA the per-block scales to C_scale_factors. requant.cpp only ever does a
copy-OUT of the fp8 (it never re-feeds a gemm and never relies on C_scale_factors for a
subsequent matmul), so the scale-for-chaining path is unexercised by any reference.
NEEDS (RTL-owner): how does a chained gemm obtain the requantizer's per-block scales?
  - Are they DMA'd to the configured GMEM addr automatically, or only on a separate
    command / gemm move-out? Is there a scalefac->PE config the next gemm needs?
  - Does SKIP_A (skipping the A move-in) also skip loading A_scales into the SF SMEM,
    so the mesh runs with stale/zero scales? (Likely related: load_scale_factors must
    still run for A, AND the requant scales must be where it reads.)
Everything else works: QK^T 0.89%, softmax P 0.74%, O-extraction path lands all cells,
requant emits fp8. The lone gap is wiring the requant's per-block scales into PV.

## (earlier, partly superseded) requant full-range elements vs MX mesh
gemmini_fence() after the requant (to drain the scale DMA) did NOT fix it -> O still
saturates uniformly to 0x7e40 (= a near-max bf16 ~6.8e37, i.e. the mesh OVERFLOW/
saturate value, not small garbage). So the cause is numeric, not a scale-drain race:
the MxRequantizer emits OCP-full-range fp8 elements (~52); the PV mesh accumulates
element-products (~2700) which overflow its small exp-4 accumulator BEFORE the per-block
E8M0 scale is applied (the mesh applies the tile scale AFTER accumulation -- confirmed
by the golden model that matched QK^T at 0.89%). QK^T worked only because its operands
were O(0.05) elements. => Re-feeding the HW requantizer's output to the MX mesh is
numerically non-viable for this data on this (read-only) silicon, UNLESS the requantizer
can be configured to emit small (O(1)) elements. This needs RTL-owner input:
  Q: can MxRequantizer target a smaller output element exponent (so chained elements
     stay mesh-safe), or is OCP emax=8 fixed? If fixed, chaining requant->mesh overflows.
Best SW alternative if not: SIMT e4m3 requant (my validated encoder emits O(1) elements)
-> P fp8 into the PV A-spad in tiled format via a path that avoids both sub-word SMEM
PutPartial and the SIMT-store->Gemmini-DMA fence (open: needs the A-spad tiling + a
working ordering primitive).

## BREAKTHROUGH (O-extraction corruption SOLVED) + the requant numerics blocker
Two findings from the GMEM-feed experiments:
1. The post-requant SIMT corruption was caused by feeding the requantizer from an
   **SMEM source** (copy_P_to_requant reading P_SMEM concurrent with the requant-region
   write). Feeding from a **GMEM source** (requant.cpp's copy_gmem_to_smem_simt_bf16
   pattern, warp-0) keeps post-requant SIMT healthy -- proven: a post-requant marker
   store 0xDEADBEEF now LANDS, and the O move-out now stores all 1024 cells. FIX
   applied: softmax writes normalized P (bf16 words) to P_GMEM (pre-requant SIMT, fine);
   requantizer fed from P_GMEM. Validated en route: softmax P in GMEM = 0.74% rel err.
2. NEW blocker (requant->PV numerics): the MxRequantizer produces **full-range (OCP)
   fp8 elements** (e.g. code 0x65 ~= 52) -- NOT the O(1) elements the MX mesh needs
   (my golden uses 0x16 ~= 0.05). Fed to the PV mesh (SKIP_A reads requant output at
   spad0), these large elements OVERFLOW the mesh's small (exp-4) accumulator ->
   O saturates uniformly to 0x7e40. This is exactly the "mesh needs O(1) elements"
   constraint from the golden model -- the HW requant's element scaling is incompatible
   with directly re-feeding the MX mesh.
STATUS now: QK^T (0.89%), softmax P (0.74%), O-extraction path WORKING (stores land);
requant runs and emits varied fp8 + scales. The lone remaining blocker is the requant
output element-range vs mesh-input-range mismatch.
NEEDS (RTL-owner input): does the MxRequantizer have a config to target a SMALLER
output element scale (so chained elements stay O(1))? If not, chaining requant->mesh
overflows for this data. Alternatives to try: (a) SIMT e4m3 requant (my validated
encoder produces O(1) elements) writing P fp8 to GMEM, then PV DMA -- but that hits the
SIMT-store->Gemmini-DMA ordering (no working GMEM fence; plain fence stalls); (b) a
Gemmini-DMA move of the requant output to GMEM (asserts PutPartial currently); (c)
pre-scale P so the requant's full-range output maps to mesh-safe magnitudes.

## MARKER TEST (decisive): post-requant warp execution-state is corrupted (older)
A fixed-address constant store `*0x40050000 = 0xDEADBEEF` placed right after the
requant write (copy_P_to_requant) produces NOTHING in the trace -- 0xDEADBEEF never
appears as store data OR even as a loaded constant, and no store hits 0x40050000.
So it's not just "store address corrupted" -- the muon warp cannot even materialize a
constant after the GEMMINI_REQUANT write. This is whole-warp execution-state corruption
triggered by writing the requantizer SMEM region. The Radiance design is READ-ONLY
(taped out), so the resolution is either (a) the *correct requantizer usage protocol*
that doesn't trip this (requant.cpp only ever does a SIMT copy-OUT immediately after,
never general compute -- that may be the only validated usage), or (b) confirmation
from RTL owners that SIMT-after-requant is unsupported on silicon.
DECISIVE NEXT EXPERIMENT (recommended, ~1 build + 1 run): build requant.cpp as a
standalone ELF (PROJECT=requant in gemm_mxgemmini, or a tiny kernel dir) and check
whether ITS post-requant SIMT copy-out writes correct C_quant to GMEM. If it works ->
SIMT-after-requant IS supported and my usage differs (study the exact diff: GMEM source
vs my SMEM source for the requant-region copy; immediate copy-out vs my PV-in-between).
If it fails too -> the pattern is unsupported on this silicon and FA must extract O via
a non-SIMT, non-asserting path (fix the Gemmini-DMA move-out's sub-word SMEM access).

## FINAL ROOT CAUSE (HW requantizer ↔ SIMT): O cannot be extracted post-requant
After ~10 controlled VCS experiments the full picture is definitive:
- QK^T gemm (S, SMEM) and SIMT softmax (l) are VALIDATED on RTL (0.89% / 0.56%).
- The HW requantizer ACCEPTS P bf16 written to GEMMINI_REQUANT (0x40000), but doing so
  **breaks ALL subsequent SIMT stores**: they execute (PC reached) but compute a
  GARBAGE base register, so nothing lands at the intended GMEM address. Proven with
  BOTH the register-heavy normalize AND a trivial copy_smem_to_gmem_simt -> 0 stores.
- The only non-SIMT way to extract O (Gemmini DMA move-out, copy_C_smem_to_gmem_dma_sync)
  **asserts PutPartial** on shared_mem xbar (RadianceSharedMemComponents.scala:60,
  TLMonitor_701.sv:322) -- a sub-word SMEM access from the DMA (its "DRAM stride wrong
  for requantized output" TODO).
=> With the HW requantizer fed via GEMMINI_REQUANT, there is currently NO working path
   to move the post-PV O tensor out of SMEM in this kernel: SIMT stores are corrupted,
   Gemmini DMA asserts. This is the open blocker -- a requantizer↔SIMT/SMEM RTL
   interaction, NOT a kernel-logic bug (the compute pipeline QK->softmax->requant->PV
   all runs; folding 1/l into softmax so PV yields final O is implemented & correct).
RECOMMENDED RESOLUTION (needs RTL-owner input or FSDB warp-RF trace):
  1. Confirm with RTL owners whether SIMT compute is even SUPPORTED after writing
     GEMMINI_REQUANT, and what completion/reset handshake is required (requant.cpp only
     does a SIMT copy-OUT after, never general SIMT; that path may itself be unvalidated
     on silicon). Likely a missing drain/reset of the requantizer or a SIMT<->requant
     resource conflict.
  2. FSDB-trace muon core RF writeback at the failing post-requant store; find which
     instruction produces the garbage base and tie it to the GEMMINI_REQUANT side effect.
  3. Fix the Gemmini-DMA move-out's stride so it does full-word SMEM reads (no PutPartial)
     -- then O can be DMA'd out (Gemmini-ordered) without any post-requant SIMT.
Current kernel state: fold-1/l softmax + SIMT copy-out -> builds, runs to clean $finish,
but O=0 (post-requant SIMT corruption). git NOT pushed (per user).

## ROOT CAUSE ISOLATED: GEMMINI_REQUANT writes corrupt subsequent SIMT (older notes)
Through 6 controlled VCS experiments (32x32x32), the exact breaking op is
`copy_P_to_requant` -- the SIMT writes of P bf16 to the requantizer SMEM region
GEMMINI_REQUANT (0x40000). After it, the SIMT `normalize_output` runs (its PC is
reached) and even computes CORRECT row addresses (0x40040040, 0x40040080 seen in the
trace) but the actual store (opcode 0x23) uses a GARBAGE base register (0x7eeff7f0,
0x7ecfe800 ...), so O never lands at O_GMEM (0x40040000) -> 0 O stores -> O verify
100% err. Confirmed robust: persists with all functions noinline, dummy-gemm config
(requant.cpp-style), and even through the PV gemm's gemmini_flush between copy and
normalize. Isolation chain (O stores to 0x40040000):
- iso1  softmax_to_smem -> normalize ............................. 32 addrs  OK
- iso5  softmax + dummy-RQ-gemm + barrier(6) -> normalize ........ 32 addrs  OK
- iso6  + copy_P_to_requant (no barrier7, no PV) -> normalize .... 0         BROKEN
=> the GEMMINI_REQUANT writes themselves break later SIMT store execution (a
register/warp-state corruption), independent of barriers/PV. Not stack-clobber
(linker .shared is at 0x0 but the GPU stack is elsewhere); not a verifier artifact
(normalize computes right addrs but stores to garbage regs).
NEXT (needs FSDB warp-scheduler/register trace, or the requantizer usage protocol from
the RTL owners):
  1. FSDB-trace the muon core register file / writeback around the normalize store at
     the failing time; find which instruction writes the garbage base reg and trace
     back to the GEMMINI_REQUANT store side-effect.
  2. Check whether the MxRequantizer holds a SIMT-visible resource (SMEM bank / LSU
     credit) after being fed, and whether a specific completion/reset (gemmini_fence,
     a read of spad0, or a config) is required before resuming SIMT -- requant.cpp does
     NO SIMT compute after the requant write (only a copy-out), so this path may be
     unexercised in silicon.
  3. Try: requant + PV as the final ops with NO SIMT after (write O_unnorm out via
     Gemmini DMA, do the /l normalization offline or in a 2nd kernel launch).
Kernel left in the full requant-path state (builds, runs to $finish). git NOT pushed
(per user). Validated pieces unchanged: QK^T 0.89%, softmax l 0.56%.

## HW-requantizer path IMPLEMENTED (runs to $finish; O not yet correct)
Implemented the user-directed design in flash_attention_mx.cpp + flash_mx_impl.hpp:
- `softmax_to_smem`: SIMT softmax reads S from SMEM, writes P as bf16 (packed words) to
  P_SMEM (0x10000); l in SMEM(bf16)+GMEM(fp32).
- requant: `configure_mxgemmini<RQ>` (QUANT_OUTPUT=true) then `copy_P_to_requant`
  (single warp streams P bf16 -> GEMMINI_REQUANT 0x40000); HW emits P fp8 at spad 0 +
  E8M0 scales -> C_scale_factors.
- PV gemm: `mxgemm_single_output_tile<PV,false,SKIP_A=true>` reads A=P fp8 from spad 0
  (added a SKIP_A template to copy_gmem_to_smem_async/single_output_tile -> skip_lda=1),
  A_scales=C_scale_factors, B=V. -> O bf16 at SPAD_DEST. normalize -> O GMEM.
RESULT: builds; runs to clean $finish at 32x32x32; softmax l still written (129 stores).
BUT end-to-end O is NOT produced (0 stores to O_GMEM 0x40040000) and the run is only
~52k cycles (vs ~113k for the prior SIMT-P version) -- so the **Gemmini matmuls appear
not to actually execute** under this structure (kernel races through; warps do traverse
past PV to max pc 0x10003f68). Suspects to FSDB-debug NEXT (rtl-debug skill):
  1. Does the QK gemm still run its matmul here? (verify gemmini mesh activity / S in SMEM)
  2. Does configure_mxgemmini<RQ> (QUANT_OUTPUT=true, with the dummy-gemm OMITTED -- I
     only called configure, requant.cpp runs a full dummy mxgemm_single_output_tile to
     "clear X's") leave Gemmini unable to run the subsequent PV matmul? Try running the
     full dummy gemm like requant.cpp instead of bare configure.
  3. Does SKIP_A actually keep the matmul (skip only the A *move-in*, not the compute)?
     Check matmul_tile_async still issues with skip_lda set.
  4. Why 0 O stores though normalize's PC region is reached -- is O_unnorm read / the
     store loop being predicated off? (gemmini_fence in copy/PV may have exited early.)
Fastest isolation: a standalone requant-only test (P in -> fp8 out, mvout to GMEM,
verify vs golden_P) to confirm the requant mechanism in THIS kernel before re-chaining.

## TURN-KEY CONTINUATION: HW-requantizer P path (the one remaining piece)
Current kernel state: builds + runs to clean $finish at 32x32x32 (2 warps). QK^T S
(0.89% rel err) and softmax l (0.56%) VALIDATED on RTL. The full pipeline executes;
the ONLY gap is O accuracy, caused by P being round-tripped through GMEM with no
working ordering fence before the PV gemm's DMA. Implement this to finish:

1. softmax: replace the SIMT e4m3+GMEM requant tail of `softmax_requant` with writing
   P **bf16 packed words** to an SMEM scratch P_SMEM (a `softmax_to_smem` variant was
   drafted in-conversation: `Prow[j*NT+lane] = pack_bf16x2(a,b)`); keep l in SMEM/GMEM.
   Choose P_SMEM offset clear of S(0x1000..),scratch(0xC000),l(0xE000) — e.g. 0x10000.
2. requant config: before requant, call configure_mxgemmini<RQ> with a GemmConfig
   {TILE_M=SQ,TILE_N=SK,TILE_K=SK?,FP8,QUANT_OUTPUT=true} + gemmini_mxquant_config_mvout
   (scale out = P_scales GMEM addr). (requant.cpp runs a dummy mxgemm_single_output_tile
   <QUANT_OUTPUT=true> to do this AND clear X's; safest to copy that exactly.)
3. single-warp copy P_SMEM bf16 -> GEMMINI_REQUANT (0x40000), sequential ascending
   (drafted `copy_P_to_requant`); then mu_fence_smem(); barrier. HW emits P fp8 at
   SMEM 0x0 (= A scratchpad even, calculate_spad_addr<false>(0)==0) + E8M0 scales to
   the configured GMEM addr.
4. PV gemm reading P from spad 0x0 with **skip A move-in**: add a skip_lda path to
   copy_gmem_to_smem_async / mxgemm_single_output_tile (loop_matmul_skips(skip_lda=1,
   skip_ldb=0,...)) so it DMAs only B=V; A is the requant output already in spad 0.
   A_scales = the requant scale GMEM addr; B_scales = V_scales. -> O bf16 at SPAD_DEST.
   This whole chain is Gemmini-DMA-ordered (gemmini_fence works); NO plain `fence`.
5. normalize (done) -> O bf16 to GMEM; verify with fa_verify_out.py --base 0x40040000
   --elem-bytes 2 --golden golden_O_u16.npy. Target: O rel err ~ golden MX (~tens of %
   vs fp32 is expected for e4m3 attention; vs golden_O should be small).
Then scale 32->128x128x64: PV is non-square (N=d=64 != M=128) -> relax the
TILE_M==TILE_N static_assert or pad d to 128. Bump warps for speed once stable.
Pitfalls already mapped: plain mu_fence stalls; SIMT sub-word GMEM/SMEM stores are
unreliable (use packed words / HW requant); read gemm results from SMEM not GMEM;
tmask in .out is char-per-lane; verify via clean bf16 word-store O.

## (superseded) original FA-K4 plan: SIMT online softmax + P requant to fp8
On S (bf16): *softmax_scale, rowmax m, P=exp(S-m), rowsum l, requantize P to fp8 e4m3 +
E8M0 group-32 scales in the PV A-operand layout. Reductions via warp-cooperative pattern
(see flash_impl.hpp rowmax). Use 32-bit sw.shared for smem (avoid PutPartial assert).
Open question: expf availability in muon libc (else polynomial). Verify P_u8/Pscales/m/l
vs golden_*.npy. Then FA-K5: PV gemm O=P@V + normalize by l.

---
## Garden port notes (Agus, 2026-07-15)
- Fetched to `radiance-kernels/kernels/flash_attention_mx_yrh/`. Builds CLEAN on tapeout-330 (dropped
  the external gemmini_host_shim.h; our headers suffice). Runs clean on our cyclotron (185K cyc) AND
  Verilator RTL (243K cyc, $finish, NO l0d assert) at 2 warps.
- **Correctness in our env:** `fa_verify_out.py` expects the VCS `.out` [ISSUE] text trace; our
  cyclotron/Verilator emit a sqlite trace-db. Wrote **`fa_verify_sqlite.py`** (reconstructs O from the
  trace-db `inst` table; MUST filter to store PCs via --elf or non-store instrs with in-range rs1
  corrupt it). Validated: reproduces the kernel's actual state.
- **The fetched .cpp is the SERIAL-ISOLATION WIP (O rel err 30.3% vs golden_O_flash), not the validated
  2.48% version.** This matches the inline comment "if 34% -> bank0-KV corrupts the matmul" — i.e. an
  open bank-layout correctness bug, independent of overlap. So the overlap flip is premature: the kernel
  needs its bank-KV correctness fixed first (Richard's active debugging). Eval: MX util 6.7%/whole,
  SIMT-softmax-bound, overlap ~1.0x — headroom is real once correctness lands.
