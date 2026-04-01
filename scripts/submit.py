"""
Slurm job submission helper.

Wraps any experiment script in a dynamically-generated sbatch job.
The generated .sbatch file is saved alongside the job output for reproducibility.

Usage:
    python scripts/submit.py scripts/run_eva.py \\
        --model google/gemma-3-4b-it --dataset vqav2 --n_samples 5000 --resume \\
        -- \\
        --partition gpu --gpus 1 --time 10:00:00 --mem 48G

Everything before '--' is passed to the Python script.
Everything after  '--' is a slurm option (use long names without the leading --).

Slurm options (all optional, have defaults):
    partition   Slurm partition name             (default: gpu)
    qos         Slurm QOS                        (default: gpu_normal)
    gpus        Number of GPUs                   (default: 1)
    gpu_type    GPU constraint, e.g. a100        (default: none)
    time        Wall time limit                  (default: 12:00:00)
    mem         Memory per node                  (default: 48G)
    cpus        CPUs per task                    (default: 4)
    job_name    Job name shown in squeue         (default: inferred from script name)
    output_dir  Where to write .out/.err/.sbatch (default: results/logs/)
    env_setup   Shell command to activate env    (default: see CLUSTER_ENV_SETUP below)

Examples:
    # Exp 2.1 on VQAv2
    python scripts/submit.py scripts/run_eva.py \\
        --model google/gemma-3-4b-it --dataset vqav2 --n_samples 5000 --resume \\
        -- partition=gpu time=10:00:00 mem=48G

    # Exp 2.5 SAE convergence
    python scripts/submit.py scripts/run_sae_convergence.py \\
        --model google/gemma-3-4b-it --model_size 4b --n_samples 500 --width 64k --resume \\
        -- partition=gpu time=6:00:00 mem=64G

    # Dry run (print sbatch file without submitting)
    python scripts/submit.py scripts/run_eva.py --dataset vqav2 -- partition=gpu --dry_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from datetime import datetime

# ---------------------------------------------------------------------------
# Cluster-specific defaults
# ---------------------------------------------------------------------------
CLUSTER_ENV_SETUP = (
    # Prefer local uv/venv environment first, then conda env, otherwise fail clearly
    "if [ -f .venv/bin/activate ]; then\n"
    "    source .venv/bin/activate\n"
    "elif [ -n \"$CONDA_DEFAULT_ENV\" ]; then\n"
    "    conda activate $CONDA_DEFAULT_ENV\n"
    "else\n"
    "    echo \"No environment found (.venv missing and CONDA_DEFAULT_ENV not set).\" >&2\n"
    "    exit 1\n"
    "fi"
)
DEFAULT_PARTITION = "gpu"
DEFAULT_QOS = "gpu_normal"
DEFAULT_GPUS = 1
DEFAULT_TIME = "12:00:00"
DEFAULT_MEM = "48G"
DEFAULT_CPUS = 4
# ---------------------------------------------------------------------------


def parse_slurm_opts(slurm_args: list[str]) -> dict:
    """Parse key=value or bare flag slurm options from post-'--' args."""
    opts = {}
    for arg in slurm_args:
        arg = arg.lstrip("-")   # tolerate leading -- if user adds them
        if "=" in arg:
            k, v = arg.split("=", 1)
            opts[k.strip()] = v.strip()
        else:
            opts[arg.strip()] = True
    return opts


def build_sbatch(
    script_path: str,
    script_args: list[str],
    slurm_opts: dict,
    repo_root: str,
    log_dir: str,
    job_name: str,
) -> str:
    """Return a complete sbatch script as a string."""
    partition = slurm_opts.get("partition", DEFAULT_PARTITION)
    qos = slurm_opts.get("qos", DEFAULT_QOS)
    gpus = slurm_opts.get("gpus", DEFAULT_GPUS)
    gpu_type = slurm_opts.get("gpu_type", "")
    time = slurm_opts.get("time", DEFAULT_TIME)
    mem = slurm_opts.get("mem", DEFAULT_MEM)
    cpus = slurm_opts.get("cpus", DEFAULT_CPUS)
    env_setup = slurm_opts.get("env_setup", CLUSTER_ENV_SETUP)

    gres_line = f"#SBATCH --gres=gpu:{gpu_type}:{gpus}" if gpu_type else f"#SBATCH --gres=gpu:{gpus}"

    script_args_str = " ".join(f'"{a}"' if " " in a else a for a in script_args)

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --partition={partition}
        #SBATCH --qos={qos}
        {gres_line}
        #SBATCH --time={time}
        #SBATCH --mem={mem}
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --output={log_dir}/{job_name}_%j.out
        #SBATCH --error={log_dir}/{job_name}_%j.err

        set -euo pipefail
        echo "Job $SLURM_JOB_ID started at $(date)"
        echo "Node: $SLURMD_NODENAME"
        echo "GPUs: $CUDA_VISIBLE_DEVICES"

        cd {repo_root}

        # Activate environment
        {env_setup}

        python {script_path} {script_args_str}

        echo "Job $SLURM_JOB_ID finished at $(date)"
    """)


def main():
    # Split sys.argv at '--' to separate script args from slurm opts
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        script_and_args = argv[:split]
        slurm_raw = argv[split + 1:]
    else:
        script_and_args = argv
        slurm_raw = []

    if not script_and_args:
        print(__doc__)
        sys.exit(1)

    target_script = script_and_args[0]
    script_args = script_and_args[1:]
    slurm_opts = parse_slurm_opts(slurm_raw)
    dry_run = slurm_opts.pop("dry_run", False)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Infer job name from script name + timestamp
    script_stem = os.path.splitext(os.path.basename(target_script))[0]
    timestamp = datetime.now().strftime("%m%d_%H%M")
    job_name = slurm_opts.pop("job_name", f"{script_stem}_{timestamp}")

    log_dir = os.path.join(repo_root, slurm_opts.pop("output_dir", "results/logs"))
    os.makedirs(log_dir, exist_ok=True)

    sbatch_content = build_sbatch(
        script_path=target_script,
        script_args=script_args,
        slurm_opts=slurm_opts,
        repo_root=repo_root,
        log_dir=log_dir,
        job_name=job_name,
    )

    sbatch_path = os.path.join(log_dir, f"{job_name}.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(sbatch_content)

    print(f"Generated: {sbatch_path}")
    print()
    print(sbatch_content)

    if dry_run:
        print("(dry run — not submitting)")
        return

    result = subprocess.run(
        ["sbatch", sbatch_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"sbatch failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)

    print(result.stdout.strip())   # "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]

    # Rename the sbatch file to include the job ID for traceability
    final_sbatch = os.path.join(log_dir, f"{job_name}_{job_id}.sbatch")
    os.rename(sbatch_path, final_sbatch)
    print(f"Saved: {final_sbatch}")
    print(f"Logs:  {log_dir}/{job_name}_{job_id}.{{out,err}}")


if __name__ == "__main__":
    main()
