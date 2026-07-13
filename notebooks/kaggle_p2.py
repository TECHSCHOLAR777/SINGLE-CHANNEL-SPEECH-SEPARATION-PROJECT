# %% [markdown]
# # CA-MoSE — full P2 + P3 pipeline (Kaggle T4, Run All)
#
# **This notebook runs top-to-bottom with `Save & Run All (Commit)`.** It builds
# the mixed 2–5 speaker cache, trains the cascade, and produces every gate
# artifact (P2-INT4 verdict, P1-INT2 identity lock, M3 confusion matrix +
# calibration curve).
#
# ## Before you run — 3 one-time setup steps
# 1. **Settings → Accelerator → GPU T4 ×2**
# 2. **Settings → Internet → On**
# 3. **Add-ons → Secrets → Add secret** named **`GH_TOKEN`** = a GitHub
#    fine-grained PAT with **Contents: Read** on the private repo (the repo is
#    private, so an anonymous clone fails without this).
#
# Then hit **Save & Run All (Commit)**. First run is long (cache build + two
# trainings, ~3–5 h) but everything is resumable and checkpointed under
# `/kaggle/working`, which Kaggle saves.
#
# ## What each stage closes
# | Cell | Produces | Gate |
# |------|----------|------|
# | 4    | mixed 2–5 spk frozen-expert cache | — |
# | 5    | trained cascade checkpoint | P2-INT3 |
# | 6    | cascade-vs-expert table (fusion + sr-primary) + escalation | P2-INT4/INT5 |
# | 7–8  | stop-classifier + confusion matrix + calibration curve | M3 |
# | 9    | cross-chunk identity lock on real speech | P1-INT2 |

# %% [markdown]
# ## Cell 1 — clone (private repo) + install experts

# %%
import json
import os
import subprocess
import sys

WORK = "/kaggle/working"
SRC = f"{WORK}/CA-MoSE"
SRC_REPO = f"{WORK}/SR_CorrNet_SS"
REPO_PATH = "TECHSCHOLAR777/SINGLE-CHANNEL-SPEECH-SEPARATION-PROJECT.git"
MIXED_HF_MODEL = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"  # 2–5 speaker expensive expert

# GitHub token from the Kaggle Secret (never printed / persisted to disk).
try:
    from kaggle_secrets import UserSecretsClient

    _GH = UserSecretsClient().get_secret("GH_TOKEN")
except Exception as exc:
    raise RuntimeError(
        "Missing Kaggle Secret 'GH_TOKEN'. Add-ons -> Secrets -> add GH_TOKEN "
        "(a GitHub fine-grained PAT with Contents:Read on the private repo)."
    ) from exc

_AUTH = f"https://{_GH}@github.com/{REPO_PATH}"
if not os.path.isdir(SRC):
    subprocess.run(["git", "clone", "--branch", "parv", "--depth", "1", _AUTH, SRC], check=True)
else:
    subprocess.run(["git", "-C", SRC, "remote", "set-url", "origin", _AUTH], check=True)
    subprocess.run(["git", "-C", SRC, "pull", "origin", "parv"], check=True)
# Strip the token back out of .git/config so it never sits in the saved output.
subprocess.run(
    ["git", "-C", SRC, "remote", "set-url", "origin", f"https://github.com/{REPO_PATH}"], check=True
)
del _GH, _AUTH
os.chdir(SRC)
print("cwd:", os.getcwd())

# Cheap expert (MossFormer2 via clearvoice) + ECAPA/REAL-M (speechbrain).
subprocess.run("pip install -q clearvoice speechbrain soundfile scipy tqdm", shell=True)
# Expensive expert: SR-CorrNet-SS strictly (no SepFormer), editable + HF hub extra.
if not os.path.isdir(SRC_REPO):
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/dmlguq456/SR_CorrNet_SS.git", SRC_REPO],
        check=True,
    )
subprocess.run(f'pip install -q -e "{SRC_REPO}[hub]"', shell=True)
print("install done")

# %% [markdown]
# ## Cell 2 — LibriSpeech dev-clean → speaker-disjoint train/dev pools

# %%
import glob
import pathlib
import random

LS_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
LS_ROOT = f"{WORK}/LibriSpeech/dev-clean"
if not os.path.isdir(LS_ROOT):
    subprocess.run(
        f"cd {WORK} && wget -q {LS_URL} && tar xzf dev-clean.tar.gz && rm dev-clean.tar.gz",
        shell=True,
        check=True,
    )

speaker_dirs = sorted(p for p in glob.glob(f"{LS_ROOT}/*") if os.path.isdir(p))
random.Random(0).shuffle(speaker_dirs)
split = int(0.8 * len(speaker_dirs))
train_spk, dev_spk = speaker_dirs[:split], speaker_dirs[split:]
print(f"{len(speaker_dirs)} speakers -> {len(train_spk)} train / {len(dev_spk)} dev (disjoint)")


def link_pool(spk_dirs, name):
    """Symlink every flac of the chosen speakers into one flat pool dir."""
    pool = f"{WORK}/pool_{name}"
    pathlib.Path(pool).mkdir(parents=True, exist_ok=True)
    n = 0
    for sd in spk_dirs:
        for f in glob.glob(f"{sd}/**/*.flac", recursive=True):
            dst = os.path.join(pool, os.path.basename(f))
            if not os.path.exists(dst):
                os.symlink(f, dst)
            n += 1
    print(f"pool_{name}: {n} files")
    return pool


TRAIN_POOL = link_pool(train_spk, "train")
DEV_POOL = link_pool(dev_spk, "dev")

# %% [markdown]
# ## Cell 3 — SR-CorrNet import probe (fail fast before the slow cache build)

# %%
_probe = subprocess.run(
    [sys.executable, "-c", "from sr_corrnet import SSInference; print('sr_corrnet import OK')"],
    capture_output=True,
    text=True,
)
print(_probe.stdout.strip() or _probe.stderr.strip())
assert "OK" in _probe.stdout, "SR-CorrNet failed to import — check cell 1 install"
print("expensive expert:", MIXED_HF_MODEL)

# %% [markdown]
# ## Cell 4 — build the MIXED 2–5 speaker cache (slow, resumable, one-time)
# Smoke sizes 500 train / 100 dev. Resume is automatic — re-running continues
# from the last flushed shard. Bump `LIMIT_*` once the pipeline is proven.

# %%
LIMIT_TRAIN, LIMIT_DEV = 500, 100


def build_mixed(pool, out, limit):
    cmd = [
        sys.executable, "-m", "scripts.build_train_cache",
        "--dynamic-source-glob", f"{pool}/*.flac",
        "--allowed-n", "2", "3", "4", "5", "--target-speakers", "5",
        "--limit", str(limit), "--segment-seconds", "3.0",
        "--device", "cuda", "--out-dir", out, "--shard-size", "128",
        "--srcorrnet-hf-model", MIXED_HF_MODEL,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


build_mixed(TRAIN_POOL, f"{WORK}/cache_mixed/train", LIMIT_TRAIN)
build_mixed(DEV_POOL, f"{WORK}/cache_mixed/dev", LIMIT_DEV)
subprocess.run(f"cat {WORK}/cache_mixed/train/manifest.json", shell=True)

# %% [markdown]
# ## Cell 5 — train the cascade on the mixed-N cache (P2-INT3)

# %%
subprocess.run(
    [
        sys.executable, "-m", "train.trainer",
        "--cache-dir", f"{WORK}/cache_mixed/train",
        "--val-cache-dir", f"{WORK}/cache_mixed/dev",
        "--epochs", "30", "--batch-size", "8", "--device", "cuda",
        "--output", f"{WORK}/outputs/training_mixed/checkpoint.pt",
    ],
    check=True,
)

# %% [markdown]
# ## Cell 6 — P2-INT4 / INT5: cascade vs single experts (fusion AND sr-primary)
# Two tau sweeps on the SAME checkpoint (no retrain): the learned fusion, and
# the sr-primary routing (escalated -> raw SR-CorrNet, else MossFormer2). If
# `beats?` is True at any tau, P2-INT4 passes; otherwise the honest story is the
# quality/compute trade (escalation rate + Expected RTF).

# %%
CKPT = f"{WORK}/outputs/training_mixed/checkpoint.pt"


def sweep(sr_primary):
    tag = "sr-primary" if sr_primary else "fusion"
    print(f"\n=== {tag} ===")
    print(f"{'tau':>4} {'escal%':>7} {'cascade':>8} {'moss':>7} {'expensive':>10} {'beats?':>7}")
    best = None
    for tau in [6, 8, 10, 12, 16, 20, 100]:
        cmd = [
            sys.executable, "-m", "scripts.evaluate_cascade",
            "--cache-dir", f"{WORK}/cache_mixed/dev", "--checkpoint", CKPT,
            "--device", "cuda", "--tau", str(tau),
        ]
        if sr_primary:
            cmd.append("--sr-primary")
        d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
        s = d["si_sdri_db"]
        print(
            f"{tau:>4} {d['escalation_rate'] * 100:>6.0f}% {s['cascade']:>8.2f} "
            f"{s['mossformer2']:>7.2f} {s['expensive']:>10.2f} "
            f"{str(d['cascade_beats_single_expert']):>7}"
        )
        if best is None or s["cascade"] > best[1]:
            best = (tau, s["cascade"], d["cascade_beats_single_expert"])
    print(f"best {tag}: tau={best[0]} -> {best[1]:.2f} dB, beats_single_expert={best[2]}")
    return best


best_fusion = sweep(sr_primary=False)
best_srprimary = sweep(sr_primary=True)

# %% [markdown]
# ## Cell 7 — train the speaker-count stop-classifier (P3 / M3)

# %%
subprocess.run(
    [
        sys.executable, "-m", "scripts.train_stop_classifier",
        "--cache-dir", f"{WORK}/cache_mixed/train",
        "--val-cache-dir", f"{WORK}/cache_mixed/dev",
        "--epochs", "80", "--device", "cuda",
        "--out", f"{WORK}/outputs/counting/stop_classifier.pt",
    ],
    check=True,
)

# %% [markdown]
# ## Cell 8 — unknown-N counting eval → M3 artifacts
# Confusion matrix + calibration curve + ECE (CSV/SVG under outputs/counting).

# %%
subprocess.run(
    [
        sys.executable, "-m", "scripts.eval_counting",
        "--cache-dir", f"{WORK}/cache_mixed/dev",
        "--checkpoint", f"{WORK}/outputs/counting/stop_classifier.pt",
        "--output-dir", f"{WORK}/outputs/counting", "--count-range", "2", "5",
    ],
    check=True,
)
subprocess.run(f"ls -la {WORK}/outputs/counting", shell=True)

# %% [markdown]
# ## Cell 9 — P1-INT2: cross-chunk identity lock on real speech
# 2-speaker mix (MossFormer2's genuine regime) — should report `passed: true`.
# The identity-lock logic is already proven in CI; this is the real-speech pass.

# %%
subprocess.run(
    [
        sys.executable, "-m", "scripts.validate_alignment",
        "--dynamic-source-glob", f"{DEV_POOL}/*.flac",
        "--dynamic-n", "2", "--dynamic-seconds", "10.0",
        "--device", "cuda", "--skip-pair",
        "--output-dir", f"{WORK}/outputs/p1_alignment_2spk",
    ],
    check=False,  # non-strict so the JSON always prints
)
subprocess.run(f"cat {WORK}/outputs/p1_alignment_2spk/alignment_validation.json", shell=True)

# %% [markdown]
# ## Cell 10 — summary of everything (paste this back)

# %%
print("=" * 60)
print("CA-MoSE run summary")
print("=" * 60)
print(f"P2-INT4 fusion:     best cascade {best_fusion[1]:.2f} dB, beats={best_fusion[2]}")
print(f"P2-INT4 sr-primary: best cascade {best_srprimary[1]:.2f} dB, beats={best_srprimary[2]}")
_cs = json.loads((pathlib.Path(WORK) / "outputs/counting/counting_summary.json").read_text())
print(f"M3 count accuracy:  {_cs['counting']['accuracy']:.3f}")
if _cs.get("calibration"):
    print(f"M3 calibration ECE: {_cs['calibration']['ece']:.4f}")
subprocess.run(f"du -sh {WORK}/cache_mixed {WORK}/outputs 2>/dev/null", shell=True)
print("\nDone. Copy this summary + the cell 6 tables + cell 8 confusion matrix back to the team.")
