# Instructions for coding agents in this repository

Read this file before changing anything here. Part A is the project lead's standing requirement for all
research code (copied verbatim, 2026-11-17). Part B says how this repository meets it today and what is
still open; keep Part B up to date when the structure changes.

---

## Part A — Research Code Maintainability & Experiment Reproducibility Requirements

These requirements apply to all research-related coding agents and are **additional to all existing repository instructions, skills, and coding standards**.

The primary objective is to maintain a research codebase that is **minimal, modular, reusable, experiment-oriented, and reproducible**.

Agents must optimise not only for functional correctness, but also for long-term maintainability, ease of experimentation, and the ability for other agents or researchers to understand and reproduce experimental results.

### 1. Code Simplicity and File Size

- Write concise, readable, and purposeful code. Avoid unnecessary complexity, boilerplate, redundant wrappers, and excessive abstraction.
- **No manually maintained source code file should exceed 500 lines.** When a file would exceed this limit, refactor it into two or more logically cohesive modules.
- Split files according to responsibility, not arbitrarily by line count.
- Each module should have a clear and limited purpose, such as data loading, model definition, training, evaluation, configuration, or experiment execution.
- Prefer small, focused functions and modules over monolithic implementations.
- Do not sacrifice readability, correctness, or experimental transparency merely to reduce the number of lines.
- Avoid creating unnecessary files or abstractions when a simpler structure is sufficient.

### 2. Shared Modules and Code Reusability

**Any functionality that can reasonably be shared across experiments should be implemented once and reused.**

- Identify existing reusable components before writing new code.
- Centralise shared functionality, including data processing, model loading, training loops, evaluation metrics, inference utilities, checkpoint management, and logging.
- Avoid duplicating implementations across experiment-specific scripts.
- Experiment scripts should import and compose shared modules rather than reimplementing existing functionality.
- When multiple experiments require similar code with minor differences, extract the common logic and expose the differences through configurable parameters or small extension functions.
- Maintain a single authoritative implementation for each shared operation.
- Do not introduce complex inheritance structures, excessive conditional branches, or unnecessarily generic frameworks solely to eliminate a few lines of duplication.

**Prefer simple composition and reusable functions over duplicated code or over-engineered abstractions.**

### 3. Configuration-Driven Experiment Design

Experiments should be defined primarily through configuration rather than source-code modification.

- Separate experiment parameters from shared implementation logic.
- Store experiment-specific parameters in dedicated configuration files, such as YAML, TOML, or JSON.
- Parameters may include learning rates, batch sizes, datasets, model variants, random seeds, training schedules, reward settings, checkpoint paths, and evaluation options.
- Use a common base configuration when several experiments share the same settings.
- Express experimental differences as explicit configuration overrides rather than duplicating entire configurations.
- Avoid hardcoding experiment-specific values inside reusable modules.
- Support command-line configuration overrides when practical.
- Keep the configuration inheritance and override mechanism simple, explicit, and predictable.

When two experiments differ only in parameters, they should generally reuse the **same implementation and execution entrypoint**, with separate experiment configurations.

When experiments differ in actual functionality, they should reuse the common implementation while defining only the necessary experiment-specific functions.

### 4. Experiment-Specific Entry Points

Every meaningful training, testing, evaluation, or ablation experiment must have a clearly identifiable and independently executable entrypoint.

- Each experiment must have a straightforward way to execute its complete intended workflow.
- Experiment entrypoints should remain thin and primarily handle configuration loading, orchestration, and calls to reusable modules.
- Keep experiment-specific logic separate from shared functionality.
- If an experiment introduces unique behaviour, implement only that behaviour in the experiment-specific code.
- If multiple experiments share the same execution workflow, use a common runner with different configurations rather than creating redundant scripts.
- Do not require users to manually edit Python source files to switch between experiments.
- Avoid hidden dependencies on execution order, interactive sessions, or undocumented local state.

An experiment should be understandable from its entrypoint, configuration, and directly referenced modules.

### 5. Simple Command-Line Execution

**Every experiment must provide a simple, documented CLI command that reproduces its intended pipeline.**

For example:

- Training: `python -m experiments.train --config configs/experiment_a.yaml`
- Evaluation: `python -m experiments.evaluate --config configs/experiment_a.yaml`
- Ablation: `python -m experiments.train --config configs/ablation_b.yaml`
- Parameter override: `python -m experiments.train --config configs/experiment_a.yaml --learning-rate 1e-5`

These commands are illustrative; agents should adapt them to the actual repository structure.

Requirements:

- Prefer one-command execution whenever feasible.
- Use meaningful, consistent CLI arguments.
- Provide sensible defaults where appropriate.
- Allow experimental configurations to be selected without modifying the implementation.
- Ensure commands work from a clearly documented working directory, preferably the repository root.
- Document necessary prerequisites, including datasets, checkpoints, dependencies, and environment variables.
- Every experiment should have its exact execution command documented alongside its configuration or in an experiment index.

For multi-stage pipelines, provide a simple orchestration entrypoint that invokes the necessary stages in the correct order.

### 6. Reproducibility and Experiment Traceability

Each experiment should be reproducible and traceable to its specific configuration and implementation.

- Make random seeds explicit wherever relevant.
- Record the resolved configuration used for execution, including command-line overrides.
- Record relevant model, dataset, and checkpoint identifiers.
- Organise outputs, checkpoints, logs, and metrics into clearly identifiable experiment directories.
- Avoid silently overwriting results from previous experiments.
- Keep experiment identifiers and output naming consistent.
- Record the code version or Git commit when practical.
- Document non-deterministic operations or environmental limitations that prevent exact reproduction.
- Ensure the same configuration and required inputs can be used to rerun the same experimental procedure, while acknowledging that identical numerical results may depend on hardware and deterministic execution settings.

### 7. Research Repository Organisation

Prefer a structure that separates reusable implementations, experiment definitions, and configurations.

Example:

- `src/` — reusable models, datasets, training, evaluation, and utilities
- `configs/base/` — shared default configurations
- `configs/experiments/` — experiment-specific configurations and overrides
- `experiments/` — lightweight execution entrypoints and unique experimental logic
- `scripts/` — convenience commands and multi-stage pipeline orchestration
- `tests/` — unit tests and lightweight integration checks
- `outputs/` — experiment-specific results and checkpoints
- `README.md` or `EXPERIMENTS.md` — documented commands and experiment index

Adapt this structure to the existing repository instead of introducing unnecessary directories or reorganising working code without a clear benefit.

### 8. Refactoring and Existing Code Maintenance

When modifying or extending an existing research repository:

1. Inspect the current implementation and identify reusable functionality before introducing new code.
2. Prefer extending existing shared modules over duplicating implementations.
3. Refactor repeated experiment logic into shared components when doing so improves clarity.
4. Keep experiment-specific modifications isolated and explicit.
5. Preserve existing experimental behaviour, configuration semantics, checkpoint compatibility, and output formats unless changes are intentionally requested.
6. Avoid broad architectural rewrites when a small refactor is sufficient.
7. Verify that refactoring has not unintentionally changed experiment results or execution behaviour, using appropriate tests or representative comparisons.
8. Update affected experiment commands, configuration files, and documentation whenever execution behaviour changes.

### 9. Agent Implementation Checklist

Before considering any research coding task complete, agents should verify:

- [ ] Code is concise, readable, and free from unnecessary duplication.
- [ ] Every manually maintained source file is within the 500-line limit.
- [ ] Shared functionality is implemented in reusable modules.
- [ ] Experiment-specific parameters are externalised into configurations.
- [ ] Similar experiments reuse the same implementation wherever practical.
- [ ] Each experiment has a clearly identifiable entrypoint or configuration-driven runner.
- [ ] The relevant pipeline can be reproduced using a documented CLI command.
- [ ] Inputs, outputs, checkpoints, and experiment-specific differences are traceable.
- [ ] Existing experiments remain compatible after refactoring.
- [ ] Relevant lightweight tests or execution checks have been performed, or any unverified behaviour is explicitly reported.

**Final principle:**

> A new experiment should ideally require only a new configuration file and a simple execution command. Additional code should be written only when the experiment introduces genuinely new functionality. Shared logic should remain centralised, and experiment-specific differences should always be explicit, minimal, and easy to reproduce.

---

## Part B — How this repository maps onto Part A

| Part A | Here |
| --- | --- |
| `src/` | `affordance_vla/` (plan, heads, training, sim, analysis, legacy) |
| `configs/` | `configs/` (YAML; `base:` names the file a config extends, keys given later win) |
| `experiments/`, `scripts/` | `scripts/<purpose>/` entrypoints; `scripts/local/` for the Mac simulator |
| `outputs/` | `runs/` (on the server a link into the NAS work dir; git-ignored) |
| experiment index | `EXPERIMENTS.md` |

Rules specific to this repository:

- Code is edited and committed on the server working copy (`~/affordance-vla` on estar); a Mac clone pulls.
  Large data, checkpoints and run outputs live on the NAS (`/mnt/nas/xinyizheng/affordance_vla/`), never in git.
- GPU work goes through the shared queue (`scripts/server/submit_queue_job.sh`), never started directly.
  Since 2026-11-20 this project submits with `cards=7` (GPU 7 first, to limit how many cards it uses) and may use
  priority 164.
- Every refactor of code that produced a reported number is checked against the code before it (same inputs,
  byte-identical logs where the computation is deterministic), as `affordance_vla/sim/episode.py` was.
- Notebooks under `report/` are generated by `scripts/report/build_*.py` and executed before they are committed;
  their media go to the tracked `report/media/`.
- Reporting (since 2026-11-20): one weekly review per Europe/London calendar week (Monday to Sunday),
  `report/<Monday>_weekly_review.ipynb`, built by `scripts/report/build_weekly_<Monday as YYYY_MM_DD>.py`. Within
  the week, newer content replaces older content in the same notebook (a new pipeline figure replaces the old
  one; what changed is recorded once, in the review's change section). Notebooks of single experiments go to
  `report/experiments/` as appendices of their week and are linked from that week's review.

Still open (2026-11-17): training and evaluation runs are configured through environment variables of
`scripts/vla/run_*_gpu.sh` and `scripts/eval/run_libero_pro_gpu.sh`, documented per run in `EXPERIMENTS.md`, not
yet through `configs/`. Moving them to configuration files needs a check that every past run resolves to the
same command line.
