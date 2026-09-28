# Notebooks: which one to run

## ▶ Run next

Nothing right now. Claude will add a notebook here when a GPU run is needed.

## Done: no need to run again

| Notebook | What it did | Result |
|---|---|---|
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
