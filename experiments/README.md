# Experimental Tracking Protocol

## IndustrialDefect AI

Every experiment must have a unique reference. CAE runs are numbered separately for each category, starting at `EXP-001`. Store each record in `experiments/runs/cae-<category>/EXP-NNN.json` and set its `experiment_id` to the filename without the extension. Use the folder and ID together when citing a run, for example `cae-bottle/EXP-001` or `cae-hazelnut/EXP-001`.

## Rules

1. Each experiment has its own JSON metadata file.
2. Record model architecture and hyperparameters.
3. Record the dataset category and data split.
4. Record the Git commit corresponding to the executed code.
5. Save important metrics and visualizations.
6. Save model artifacts in persistent Kaggle Outputs.
7. Reference the Kaggle version used for the experiment.
8. Do not use test results to optimize hyperparameters or thresholds.
9. Document failed experiments and their causes.
10. Never commit large model checkpoints or datasets.

## Storage

- experiments/templates/: Experiment templates
- experiments/runs/: Experiment metadata
- notebooks/: Executable experiments
- Kaggle Outputs: Model artifacts and large results

## Reproducibility

Experiments must be reproducible from the documented
configuration, dataset version, code version and environment.

The Git commit and Kaggle version must correspond to
the code actually executed.
