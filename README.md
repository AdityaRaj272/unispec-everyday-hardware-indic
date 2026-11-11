# Training-Free Speculative Decoding on Untested Hardware and Indian Languages

Course project, DA4022 Hardware-Aware Deep Learning — Pranjal Kirar, Aditya Raj, Khushal Nagwal,
Babita Verma, Yash Pratap Singh.

Large language models write text one token at a time, and each token costs one full pass through
the network. **Speculative decoding** guesses several next tokens cheaply and lets the model check all
of them in a single pass, keeping the ones it agrees with. The output is *identical* to normal
decoding; only the speed changes.

How many tokens to guess (the **draft size g**) is a trade-off:

```
speed  =  τ(g) / s(g)
          τ(g): tokens gained per pass   – grows with g, then levels off; the same on every device
          s(g): seconds per pass          – grows with g; how fast depends on the hardware
```

Two recent papers turn this into a **hardware-aware rule**: measure the device, pick the g that
maximises τ/s.

* **UniSpec** (Do et al., ACL 2026) — training-free n-gram drafting with *device-aware calibration*:
  fit regression curves to τ(g) and s(g) on the target device, maximise their ratio with Differential
  Evolution. Evaluated on four data-centre/workstation NVIDIA GPUs and seven non-Indian languages.
* **PPD** (Chen et al., EMNLP Findings 2025) — *hardware-aware tree pruning* with the same rule,
  splitting it into a hardware-independent part (acceptance, measured once) and a hardware-dependent
  part (forward latency, profiled per device).

Both papers name the same open gaps: weak or unstable hardware, and morphologically rich languages.
This project tests the shared rule there:

| Question | Experiment |
|---|---|
| How differently do a laptop CPU and a GPU charge for checking extra tokens? | `profile` — the *hardware fingerprint* s(n) |
| Is τ(g) really device-independent? | `sweep` on every device, compare τ curves |
| Does UniSpec-style calibration find the best g on cheap hardware? Does PPD's "measure τ once" shortcut? | `calibrate` — grid search vs UniSpec fit vs PPD split, checked on held-out prompts |
| What does a g tuned for another device cost? Can it fall below 1.0×? | `calibrate` + `plots` (cross-device table) |
| Do Hindi, Marathi, Bengali, Tamil behave like English? | `language` — tokens per word, τ, speedup |
| Does calibration survive a laptop heating up or changing power mode? | `thermal`, and any step with `--tag power-saver` |

## Setup used

* Model: `Qwen/Qwen2.5-0.5B-Instruct` (small enough for any laptop CPU; ungated)
* Drafter: n-gram / prompt-lookup (training-free, no extra memory) — the drafter family UniSpec builds on
* Task: summarise a news article (XL-Sum, BBC) — a *grounded* task where the output re-uses input
  phrases. 40 articles per language, cut to 250 words, fixed in `data/xlsum_subset.jsonl`
* Greedy decoding, 128 new tokens, exact token-for-token check against normal decoding
* Precision: fp32 on CPU, fp16 on T4 (chosen automatically per device)

## Results

All figures and tables are regenerated from `results/` by `python run.py plots`;
see **[results/SUMMARY.md](results/SUMMARY.md)** for every number.

| | |
|---|---|
| ![](figures/fig1_hardware_fingerprint.png) | ![](figures/fig2_draft_size_sweep.png) |
| ![](figures/fig3_wrong_g_cost.png) | ![](figures/fig4_calibration_methods.png) |
| ![](figures/fig5_language.png) | ![](figures/fig6_thermal.png) |
| ![](figures/fig7_copying_vs_speedup.png) | |

### Findings so far (laptop CPU: Intel i7-8550U, fp32 · free GPU: Tesla T4, fp16)

* **Lossless:** 100% identical output on the CPU. On the T4, 100% in the correctness check and 90–97%
  across the sweep; the rest diverge at fp16 near-ties between two almost equally likely tokens
  (a rounding effect of half precision, not an error in the method).
* **Hardware fingerprints are opposite:** on the T4 a pass costs ~25 ms for 1 token and ~29–30 ms for
  anything from 2 to 33 tokens — checking guesses is almost free. On the CPU a 2-token pass already
  costs 1.55× a 1-token pass and a 33-token pass 3.8×.
* **τ really is device-independent:** on the 12 articles run on both devices, tokens gained per pass
  agree to two decimals at every g (e.g. 2.60 vs 2.60 at g = 8) — the assumption behind both
  UniSpec's calibration and PPD's "measure acceptance once" split.
* **So the best draft size differs:** the CPU peaks at **1.62× around g = 8–16** and declines after
  (1.44× at g = 32; on the least-copying articles g = 32 is slower than no guessing, 0.86–0.98×).
  The T4 keeps improving up to the largest g tried: 3.17× at g = 32 and, in a second sweep reaching
  g = 128 (`__wide-grid`), **4.10× at g = 128** — a 129-token pass still takes ~28 ms. With a 0.5B
  model the T4 never runs out of spare capacity in this range.
* **Cost of the wrong g (held-out articles):** the CPU running the T4's g = 32 gets 1.33× instead of
  1.54× (−14%); the T4 running the CPU's g = 14 gets 2.22× instead of 2.36× (−6%).
* **Calibration:** on the CPU, measured together on unseen articles in one session, UniSpec's fit
  (g\* = 14) gives 1.48×, grid search (g\* = 12) 1.47×, the PPD split with the quick synthetic profile
  (g\* = 22) 1.42×, and the PPD split using τ measured on the *T4* (g\* = 20) 1.38× — all within 10%,
  because the CPU's speedup curve is flat between g = 8 and 22. On the T4 all methods — including the
  PPD split using τ measured on the *laptop* — pick g = 32 (the top of that grid); in the wide sweep,
  UniSpec's fit and grid search pick g = 128. The PPD split relies on one short synthetic timing
  profile, and on the shared cloud GPU a few of its readings jump from ~28 to ~40 ms; with τ measured
  on the laptop it then picks g = 7 (1.88× instead of 2.30×). All methods over-predict the speedup,
  because acceptance depends on how much each article is copied.
* **Speedup follows copying:** per article, generation-only speedup grows steadily with the share of
  the output copied from the article, on both devices (fig. 7).
* **Languages:** Qwen's tokenizer needs 4.7–10.1 tokens per word for Hindi/Marathi/Bengali/Tamil vs 1.3
  for English. Contrary to our hypothesis, tokens gained per pass are *higher* (the 0.5B model copies
  the article heavily, and each copied word is many tokens). On the T4 this gives **5.6–8.6× end to
  end** (Hindi 8.55×). On the CPU the 3–7× longer prompts take long to read, which guessing cannot speed
  up, so end-to-end gains shrink to 1.42–1.90× although generation alone is 2.7–3.9× faster.
* **Sustained load:** over 20 minutes of continuous load the plugged-in laptop showed no throttling
  trend; the calibrated g kept its speedup (1.45–1.50× per cycle of prompts).

## Run it

Python 3.9+.

```bash
pip install -r requirements.txt
python tests/test_engine.py          # engine correctness, no model needed, < 1 s
python run.py all                    # every benchmark on this machine (laptop CPU: ~1.5 h)
python run.py thermal --minutes 20   # optional: sustained-load test (laptops)
python run.py plots                  # figures + results/SUMMARY.md
```

* The device (NVIDIA GPU, Apple Silicon GPU, or CPU), its precision and the workload are picked
  automatically. Results go to `results/<device-name>/`.
* Finished steps are skipped on a re-run, so an interrupted run can simply be restarted
  (`--force` redoes them).
* `python run.py all --preset quick` is a few-minute smoke test.
* A different condition on the same machine (e.g. battery saver) gets its own folder:
  `python run.py profile calibrate --tag power-saver` (re-profiles the hardware and re-uses the acceptance numbers already measured; a laptop on battery is detected and labelled `on-battery` automatically).
* Larger draft sizes than the default 32: `python run.py profile sweep calibrate --tag wide-grid --grid 1 2 4 8 16 32 48 64 96 128`.
* **Free T4 GPU:** open `notebooks/colab_t4.ipynb` in Google Colab with a T4 runtime and run all cells.
* For timing quality, keep other heavy programs closed while it runs and keep a laptop plugged in
  (unless that is the condition being measured).

### Adding a new device

Run the commands above on the new machine and copy its `results/<device-name>/` folder into this
repo's `results/`. `python run.py plots` then includes it in every figure. Re-running
`python run.py calibrate --force` on a device afterwards also adds the "τ measured on another
device" (PPD split) comparison.

## Code layout

```
specdec/
  engine.py       lossless draft-and-verify loop (KV-cache invariant), per-pass timing
  drafters.py     n-gram drafter; draft_len = g
  calibrate.py    grid search, UniSpec regression + Differential Evolution, PPD split
  device.py       device / precision detection, hardware description, model loading
  data.py         prompts
  experiments.py  correctness, profile, sweep, calibrate, language, thermal
  plots.py        figures and results/SUMMARY.md
run.py            command-line entry point
scripts/prepare_data.py   rebuilds data/xlsum_subset.jsonl from XL-Sum
tests/test_engine.py      mock-model tests: output must equal greedy decoding
notebooks/colab_t4.ipynb  free-GPU runner
```

The decoding engine and its tests are adapted from our earlier RetroSpec engine
(`github.com/lxzy8/RetroSpecS`), whose 7-method T4 comparison was reported at the mid-project stage.

## Scope and limitations

* A single chain of guesses, not UniSpec's confidence-scored draft trees; PPD's trained prompt
  tokens are not reproduced (they need GPU training). Both papers are used for their shared
  hardware-aware calibration rule.
* A 0.5B model, far smaller than the 7–14B models in the papers; its summaries in Indian languages
  are of low quality. Decoding is lossless, so this affects what is generated, not the validity of
  the speed comparison.
* Laptop timings are noisier than server timings; every speedup is a paired, per-prompt ratio
  measured back to back.

## References

* D.-T. Do, N.-K. Le, N. L. Minh. *UniSpec: Training-Free Speculative Decoding for Robust LLM
  Acceleration Across Languages and Hardware.* ACL 2026.
* H. Chen et al. *Hardware-Aware Parallel Prompt Decoding for Memory-Efficient Acceleration of LLM
  Inference.* Findings of EMNLP 2025.
* T. Hasan et al. *XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages.*
  Findings of ACL 2021. (Data subset used under CC BY-NC-SA 4.0.)
