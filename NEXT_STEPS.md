# Next steps: getting to a full match with little manual work

Goal for the next session: detection, tracking, ball and pitch keypoints ready, so that a full
night match can be processed end to end with manual work limited to **pass tagging**
(`notebooks/tag_passes_v2.ipynb`). Re-ID is discussed after that.

## Where we are (all committed)

| Part | Now | Evidence |
|---|---|---|
| People | our open RF-DETR Medium (Apache 2.0), cut-off 0.50 | night IDF1 0.96, day 0.82 (`data/people_model/`) |
| Tracking | one tracker for all people, role = majority vote per track | `ROLE_BY_TRACK=true` |
| Ball | our tile-trained RF-DETR Small, fast search around the ball | right in 93-96% of frames, boots 3% (`data/ball_model/`) |
| Pitch keypoints | Roboflow v10 weights run locally (Apache-2.0, no credits), plain frames + contrast stretch | night clip: 0.78 m median vs hand-placed pitch points (hosted on enhanced frames: 2.18 m) |
| Speed (T4, per frame) | people FP16 28 ms (plain 36, 99.6% same boxes); ball TensorRT 24 ms around the ball / 232 ms full search (plain 94 / 779, 100% same) | `speed_and_new_footage.ipynb` |
| Teams | colour classifier, fixed: day 98%, night 97% right; IDs unaffected | `data/team/results.json`; goalkeepers still without team |

`baseline.env` is the pilot setup. Model files live in `MyDrive/Playbook/{people_model,ball_model}/`.

## Tomorrow, GPU (you run; see `notebooks/README.md` for the steps)

1. ~~`speed_and_new_footage.ipynb`~~ done (see below).
2. ~~`train_people_rfdetr_large.ipynb`~~ done. Decision: keep Medium; Large fixes day referees (80->99%)
   but player IDs are no better (also at cut-off 0.40) and it is 2x slower (`data/people_model/large_vs_medium.json`).
3. `train_field_keypoints.ipynb` (~2-5 h): our own pitch keypoint model. Round 1 = dataset version 11
   (the 476 old labelled frames, Roboflow's split: same data as the current model v10, so a
   like-for-like test of whether our open model can replace it). Round 2 after the 120 match frames
   are labelled (needs a PC): new version, split by match, notebook regenerated for it.
   First round-1 run (field_model/v11) is invalid: Roboflow names points "01".."09", the dataset tool
   looked for "1".."9", so points 1-9 were dropped from training (fixed + tested; the notebook now
   stops if any point has no labels). Run 2 (768 px, 80 epochs): 76% found, 38 px median vs the
   current model's 97% / 10 px - still improving when it stopped; run 3 continues it to 160 epochs
   (field_model/v11-run3, through the lr drop at epoch 100).

## Tomorrow, you

- Upload new test clips to `MyDrive/Playbook/video_downloads/` (night matches preferred; full
  matches or 30-60 s in-play clips; matches not in the annotation pool are the best test).
- Keypoints: label the 120 match frames by hand in the Roboflow project `football-field-detection`
  (Annotate -> job "Pitch keypoints - our matches (120)"; any split when adding to the dataset, the
  notebook splits by match); then Claude generates version 11.

## Tomorrow, Claude

1. ~~Score the new clips~~ done: `data/new_footage/proxies.json` (`tools/proxy_metrics.py`).
   Match2 clips look as good as the CVAT clips (ball 95-96%, no ball jumps, few new IDs).
   Ittifaq (far, low-quality stream, tiny players) is the weak case: ball 78-88% with jumps to
   boots/stands (8 ball switches in 20 s). Candidate fixes: bigger ball tiles/upscale for far
   cameras, ittifaq frames in the next ball training round.
2. `MODEL_ACCEL`: ball TensorRT (identical results), people FP16 (TensorRT failed on a batch-size
   bug, fixed since; re-check in the next Colab run). Set it in the full-match notebook (GPU only).
3. Compare Large vs Medium people model on both CVAT clips; switch if better.
4. Keypoints (prepared): 120 frames from our matches (10 per match, 11 matches + HILAL-HAZM) added to
   `football-field-detection`. The old model's pre-labels were too poor and were cleared; the frames
   wait unlabelled in two annotation jobs: "1) SCORE (30)" (matches 11, 12, HILAL-HAZM; label first,
   then train a baseline on the old frames) and "2) TRAIN (90)" (matches 2-10; retrain, compare).
   Old labelled frames: HILAL-HAZM 161, HILAL-AHLI 206, SAUDI 76, other 34 -> HILAL-HAZM is a seen
   match; the main score is matches 11 + 12. The old
   dataset had no frames from these matches, near-duplicates across splits and 576x576 squashing.
   Licence checked: RF-DETR Keypoint (preview) weights are Apache 2.0 (rfdetr README).
   `notebooks/train_field_keypoints.ipynb` splits by match (11+12 valid, HILAL-HAZM test), trains at
   768 px with pitch-mirror flips, scores the test frames (px error) and records keypoints on both
   CVAT clips. Then Claude compares with the old model (pitch-line check, full pipeline) and sets
   `FIELD_MODEL_PATH` if better. Old-model error on the same test frames: run it on the 10 test frames
   (hosted, ~0.01 credit) against your labels. Version 11 settings: no resize, no contrast stretching, no Roboflow
   augmentation (the notebook augments).
5. **Team classification: fixed (measured).** True team per player labelled for both CVAT clips
   (`data/cvat/*/teams.json`), scored with `tools/team_eval.py`. Before: day 99.9%, night 81% (green
   players locked into the blue team), and team on wrecked night IDs (9 -> 175 switches). Causes: kit
   colours seeded from the two most different crops of the first second; each player's team locked
   after 8 votes; single-frame team guesses vetoing ID re-links. Now (defaults): k-means seeding +
   re-clustering every 200 samples, 1.5 s majority window, no lock, no team veto in ID re-links ->
   day 98.2%, night 97.4%, IDs unaffected (`data/team/results.json`). Left: goalkeepers (team by the
   side they defend); the remaining ~2% is IDs passing between players (Re-ID).
6. Full-match notebook (see below).

## Full-match run: what it has to handle

- **Runtime.** A 90-minute match is ~135,000 frames at 25 fps. The old pipeline ran ~1 frame/s on a
  T4; even at 3-4 frames/s that is 9-12 hours, longer than a Colab session. So the full-match
  notebook must process the match in segments (e.g. 5 minutes), save each segment's results to
  Drive and resume where it stopped. Levers to cut time: TensorRT/FP16, keypoints every 2-3 frames
  (pitch smoothing covers the gaps), the ball model's fast search.
- **Output for pass tagging.** `tag_passes_v2` reads `per_frame_tracks.csv`: frame, track_id,
  display_track_id, class_id, conf, x1..y2, x_m, y_m, team_id, ball_interpolated (all still written).
  `team_id` is filled only with team classification on (`--enable-team`, colour mode): must be on
  for the full match, and fixed first (see step 5 above: it currently mixes teams).
- **What makes tagging faster.** The tagger relocates the ball by hand when it is wrong, so ball
  accuracy saves the most manual time; passer/receiver come from display IDs, so stable IDs
  (Re-ID, next topic) come second. A short list of frames where the ball is uncertain (long gaps,
  track switches) would let the tagger check those first.
- **Joining segments.** IDs must continue across segment boundaries (carry tracker state over,
  or overlap segments and match IDs in the overlap).

## After that: Re-ID (to discuss)

Measure first on the long hand-checked windows (old pipeline: IDF1 0.55 / 0.68), then re-link
lost players by pitch position + team instead of image pixels, then a football-specific
appearance model if needed.

## Keypoints: where it stands
- Default: Roboflow's v10 weights, downloaded (Apache-2.0 per Roboflow's download page) and run with the
  open rfdetr package (`FIELD_MODEL_KIND=roboflow_v10`, notebook `field_model_v10_local.ipynb`).
- Our own keypoint model (runs 1-3) was trained with wrong mirror pairs (the pitch config lists points in
  Roboflow's order; pairs were derived by index). Fixed. A retrain with the correct pairs (+ the 120 match
  frames once labelled) is optional now that v10 runs locally.
