# Notebooks: which one to run

## ▶ Run next (needs a T4 GPU), in this order

Overall plan: `NEXT_STEPS.md` in the repo root.

Before each: if the notebook is already open from an earlier run, **Runtime → Disconnect and delete
runtime**. Then Runtime → Change runtime type → **T4 GPU** → Save, and **Runtime → Run all**. Do not
edit cells. Colab Secret `ROBOFLOW_API_KEY` must be on (key icon). Tell Claude when each one finishes.

1. **`train_people_rfdetr_large.ipynb`** (about 4-7 hours): trains the Large people model. If it
   disconnects, Run all again: it resumes. Results: `MyDrive/Playbook/people_model_large/`.
   https://colab.research.google.com/github/MuwafagQ/Playbook-program/blob/claude/setup-gpu-video-testing-JhgUH/notebooks/train_people_rfdetr_large.ipynb

Other Google account: share `MyDrive/Playbook` with it and add a shortcut to `Playbook` in its My Drive.

## Done: no need to run again

| Notebook | What it did | Result |
|---|---|---|
| `speed_and_new_footage.ipynb` | FP16/TensorRT speed test + our models on new clips | `MyDrive/Playbook/new_footage/`, `data/new_footage/` |
| `train_people_rfdetr.ipynb` | trained our people model (RF-DETR Medium, open licence), now the default | `MyDrive/Playbook/people_model/` |
| `train_ball_tiles.ipynb` | trained the ball model (RF-DETR Small, open licence) | `MyDrive/Playbook/ball_model/` |
| `ball_candidates_export.ipynb` | ball model on the night clip + T4 speed test | `MyDrive/Playbook/ball_model/` |
| `thin_annotation_pool.ipynb` | picked 25 frames per match | `MyDrive/Playbook/annotation_pool_25/` |
| `upload_pool_to_roboflow.ipynb` | uploaded them to Roboflow | project `players-detection-my09y` |
| `gsr_baseline.ipynb`, `gsr_ours.ipynb` | SoccerNet GSR benchmark | `data/gsr/` |
| `benchmark_windows.ipynb` | first benchmark on the reviewed windows | `benchmark_results/` |
| `sample_frames_for_annotation.ipynb` | frame sampler for annotation | annotation pool |

## Replaced: do not use

| Notebook | Why |
|---|---|
| `nas_people_eval.ipynb` | the NAS model is now run from Claude's environment through Roboflow's hosted API (`tools/record_models.py --hosted`); no Colab needed |
| `ball_tiles_cvat.ipynb` | replaced by the dedicated ball model |

## Earlier work (other tasks)

`merge_halves`, `manual_calibration`, `annotate_video`, `apply_excel_reid`, `excel_reid`, `auto_reid`,
`reid_stitch`, `tag_passes`, `tag_passes_v2`: tracking clean-up and tagging tools from earlier sessions.
