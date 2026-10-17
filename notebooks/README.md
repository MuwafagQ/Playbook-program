# Notebooks: which one to run

## ▶ Run next: a new match (Ittifaq_f_match_video), in this order

Before each: **Runtime → Disconnect and delete runtime** if it was open before, then **Runtime → Run all**.
Do not edit cells except the match fields at the top. Tell Claude when each one finishes.

1. **Match clip cutter** (in Drive, `video_downloads/Match clip cutter.ipynb`, CPU, ~15 min): fill in the
   video link and where each half starts and ends in the video. Clips go to `video_downloads/<match>_clips/`.
2. **`full_match.ipynb`** (L4 GPU, ~6-7 hours): tracking on every clip, players joined.
   https://colab.research.google.com/github/MuwafagQ/Playbook-program/blob/claude/setup-gpu-video-testing-JhgUH/notebooks/full_match.ipynb
3. **`after_tracking.ipynb`** (L4 GPU, ~1-2 hours): cuts tracker swaps between the two teams, re-joins the
   players, reads shirt numbers, refreshes the Pass Review data.
   https://colab.research.google.com/github/MuwafagQ/Playbook-program/blob/claude/setup-gpu-video-testing-JhgUH/notebooks/after_tracking.ipynb

Then Claude builds the Pass Review app with pass suggestions and pre-filled names (tools/pass_detect.py,
tools/name_propagate.py).

Other Google account: share `MyDrive/Playbook` with it and add a shortcut to `Playbook` in its My Drive.

## Done: no need to run again

| Notebook | What it did | Result |
|---|---|---|
| `train_people_rfdetr_large.ipynb` | trained the Large people model | `MyDrive/Playbook/people_model_large/` |
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
