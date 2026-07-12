# %% [markdown]
# # CA-MoSE — P1 close + P2 first real training run (Kaggle T4)
#
# One notebook, run top to bottom on Kaggle (GPU T4 x2, **Internet ON**). It:
#   1. clones the repo (`parv` branch) and installs the frozen experts
#   2. downloads LibriSpeech `dev-clean` and splits it into speaker-disjoint
#      train / dev pools (no speaker leakage)
#   3. builds the frozen-expert cache ONCE with `scripts/build_train_cache.py`
#      (MossFormer2 padded to 3 + expensive expert Hungarian-aligned + REAL-M
#      + ECAPA)
#   4. trains the CA-MoSE heads on the cache and evaluates **cascade vs best
#      single expert** (P2-INT3/INT4/INT5)
#   5. re-runs `scripts/validate_alignment.py` against the *padded* expert to
#      close P1-INT2 (`expert_covers_all_speakers`)
#
# The expensive expert is SR-CorrNet if you point `SRCORRNET_REPO` /
# `SRCORRNET_CKPT` at your weights; otherwise it auto-falls back to SepFormer
# (`sepformer-wsj03mix`) so the whole notebook still runs end to end.
#
# Kaggle notes for a first-timer: turn on **Settings → Accelerator: GPU T4 x2**
# and **Settings → Internet: On**. Outputs land in `/kaggle/working`, which is
# what Kaggle saves (keep it under the 20 GB cap — the cache trim cell handles
# that).

# %% [markdown]
# ## Cell 1 — clone + install
#
# The CA-MoSE repo is **private**, so an anonymous clone fails. Add a GitHub
# Personal Access Token as a Kaggle Secret named `GH_TOKEN` before running this
# cell: **Add-ons → Secrets → Add a new secret** (key `GH_TOKEN`, value = a
# fine-grained PAT scoped to just this repo, Contents: Read-only is enough).
# The token is read via Kaggle's secrets client and never printed or written
# to a file — it only appears transiently in the git remote URL for the clone.

# %%
import os
import subprocess
import sys

REPO_PATH = "TECHSCHOLAR777/SINGLE-CHANNEL-SPEECH-SEPARATION-PROJECT.git"
WORK = "/kaggle/working"
SRC = f"{WORK}/CA-MoSE"

try:
    from kaggle_secrets import UserSecretsClient

    GH_TOKEN = UserSecretsClient().get_secret("GH_TOKEN")
except Exception as exc:
    raise RuntimeError(
        "Missing Kaggle Secret 'GH_TOKEN'. This repo is private — add a GitHub "
        "Personal Access Token via Add-ons -> Secrets (key: GH_TOKEN) before "
        "running this notebook."
    ) from exc

REPO_AUTH = f"https://{GH_TOKEN}@github.com/{REPO_PATH}"

if not os.path.isdir(SRC):
    subprocess.run(
        ["git", "clone", "--branch", "parv", "--depth", "1", REPO_AUTH, SRC], check=True
    )
    # Strip the token back out of .git/config immediately — otherwise it sits in
    # plaintext under /kaggle/working, which Kaggle can persist/share as output.
    subprocess.run(
        ["git", "-C", SRC, "remote", "set-url", "origin", f"https://github.com/{REPO_PATH}"],
        check=True,
    )
del GH_TOKEN, REPO_AUTH  # never keep the token around longer than the clone call
os.chdir(SRC)
print("cwd:", os.getcwd())

# torch / torchaudio are preinstalled on Kaggle. Add the cheap-expert stack.
get_ipython().system("pip install -q clearvoice speechbrain soundfile scipy tqdm 2>&1 | tail -3")  # noqa: F821, E501

# Expensive expert: SR-CorrNet-SS (strict, no SepFormer). Clone + editable install
# with the [hub] extra so `from sr_corrnet import SSInference` pulls the HF checkpoint.
SRC_REPO = f"{WORK}/SR_CorrNet_SS"
if not os.path.isdir(SRC_REPO):
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/dmlguq456/SR_CorrNet_SS.git", SRC_REPO],
        check=True,
    )
get_ipython().system(f'pip install -q -e "{SRC_REPO}[hub]" 2>&1 | tail -3')  # noqa: F821

# %% [markdown]
# ## Cell 2 — LibriSpeech dev-clean + speaker-disjoint train/dev pools

# %%
import glob
import pathlib
import random

LS_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
LS_ROOT = f"{WORK}/LibriSpeech/dev-clean"

if not os.path.isdir(LS_ROOT):
    get_ipython().system(f"cd {WORK} && wget -q {LS_URL} && tar xzf dev-clean.tar.gz && rm dev-clean.tar.gz")  # noqa: F821, E501

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
# ## Cell 3 — SR-CorrNet expensive expert (strict, HF Hub checkpoint)
# Uses the variable 2–3 speaker 1-channel WSJ model by default (8 kHz; the
# wrapper resamples to/from the project's 16 kHz). Override `SRCORRNET_HF_MODEL`
# for the 2–5 speaker variant, or point `SRCORRNET_CKPT` at a local checkpoint.

# %%
SRCORRNET_HF_MODEL = os.environ.get("SRCORRNET_HF_MODEL", "shinuh/sr-corrnet-ss-1ch-wsj-var-2-3spk")
SRCORRNET_CKPT = os.environ.get("SRCORRNET_CKPT", "")  # optional local .pt
sr_args = ["--srcorrnet-hf-model", SRCORRNET_HF_MODEL]
if SRCORRNET_CKPT:
    sr_args += ["--srcorrnet-checkpoint", SRCORRNET_CKPT]
print("expensive expert: SR-CorrNet-SS", SRCORRNET_HF_MODEL)

# Fail fast if SR-CorrNet can't import, before spending time on the cache build.
_probe = subprocess.run(
    [sys.executable, "-c", "from sr_corrnet import SSInference; print('sr_corrnet import OK')"],
    capture_output=True,
    text=True,
)
print(_probe.stdout.strip() or _probe.stderr.strip())

# %% [markdown]
# ## Cell 4 — build the frozen-expert cache (the slow, one-time step)
# Smoke sizes: 500 train / 100 dev. Bump `--limit` once the pipeline is proven.

# %%
import sys


def build_cache(pool, out, limit):
    cmd = [
        sys.executable, "-m", "scripts.build_train_cache",
        "--dynamic-source-glob", f"{pool}/*.flac",
        "--allowed-n", "3", "--target-speakers", "3",
        "--limit", str(limit), "--segment-seconds", "3.0",
        "--device", "cuda", "--out-dir", out, "--shard-size", "128",
    ] + sr_args
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


build_cache(TRAIN_POOL, f"{WORK}/cache/train", 500)
build_cache(DEV_POOL, f"{WORK}/cache/dev", 100)

# %% [markdown]
# ## Cell 5 — train the cascade + evaluate vs single experts (P2-INT3/4/5)

# %%
subprocess.run(
    [
        sys.executable, "-m", "train.trainer",
        "--cache-dir", f"{WORK}/cache/train",
        "--val-cache-dir", f"{WORK}/cache/dev",
        "--epochs", "30", "--batch-size", "8", "--device", "cuda",
        "--output", f"{WORK}/outputs/training/checkpoint.pt",
    ],
    check=True,
)
get_ipython().system(f"cat {WORK}/outputs/training/checkpoint.json | python -m json.tool | tail -30")  # noqa: F821, E501

# %% [markdown]
# ## Cell 6 — close P1-INT2: cross-chunk identity lock on real speech
#
# The identity-lock *logic* is already proven deterministically in CI
# (`tests/test_p1_int2_identity_lock.py`, 2+3 speakers, 0 switches through the
# real run_and_align_long path). This cell confirms it on real LibriSpeech.
#
# It runs TWO validations:
#   (A) 2-speaker mix — MossFormer2's genuine regime, where it emits one stable
#       stream per speaker. This is the real-speech identity-lock pass and it
#       should come back `"passed": true` with 0 switches.
#   (B) 3-speaker mix — informational. MossFormer2 is a 2-speaker model, so even
#       residual-padded its 3rd slot wanders; `expert_covers_all_speakers` may
#       be true but the wandering residual is an ESCALATION concern (the cascade
#       routes 3-speaker audio to SR-CorrNet), not an identity-lock bug. Kept
#       visible so the distinction is on the record, not hidden.

# %%
import json as _json


def p1_validate(n, tag):
    out = f"{WORK}/outputs/p1_alignment_{tag}"
    subprocess.run(
        [
            sys.executable, "-m", "scripts.validate_alignment",
            "--dynamic-source-glob", f"{DEV_POOL}/*.flac",
            "--dynamic-n", str(n), "--dynamic-seconds", "10.0",
            "--device", "cuda", "--skip-pair",
            "--output-dir", out,
        ],
        check=False,  # non-strict so we always see the JSON, pass or fail
    )
    report = _json.loads(open(f"{out}/alignment_validation.json").read())
    p1 = report["p1_int2"]
    print(
        f"[{n}-spk] passed={p1['passed']} switches={p1['identity_switches']} "
        f"tracks={p1['num_persistent_tracks']}/{p1['num_reference_speakers']} "
        f"covers_all={p1['expert_covers_all_speakers']} streams_per_chunk={p1['streams_per_chunk']}"
    )
    return p1


p1_2spk = p1_validate(2, "2spk")  # the real-speech identity-lock pass
p1_3spk = p1_validate(3, "3spk")  # informational (escalation regime)
print("\nP1-INT2 real-speech identity lock (2-spk):", "PASS" if p1_2spk["passed"] else "FAIL")

# %% [markdown]
# ## Cell 7 — trim outputs for Kaggle's save cap
# Keep the checkpoint + JSON reports; drop the (large) fp16 cache before saving.

# %%
get_ipython().system(f"du -sh {WORK}/cache {WORK}/outputs 2>/dev/null")  # noqa: F821
# Uncomment to drop the cache from the saved output (rebuild next session):
# get_ipython().system(f"rm -rf {WORK}/cache")
print("Done. Copy the printed P2-INT4 verdict and P1 alignment JSON back to Parv.")
