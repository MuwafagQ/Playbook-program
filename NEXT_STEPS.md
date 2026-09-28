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
| Pitch keypoints | Roboflow model (RF-DETR keypoint preview, not ours) | good on our footage (~0.5 m), weak on others |
| Speed | not optimised yet; TensorRT/FP16 ready to test | `vision/fast_rfdetr.py`, `MODEL_ACCEL` |
| Teams | colour classifier; mixes teams (not yet measured) | to fix tomorrow |

`baseline.env` is the pilot setup. Model files live in `MyDrive/Playbook/{people_model,ball_model}/`.

## Tomorrow, GPU (you run; see `notebooks/README.md` for the steps)

1. `speed_and_new_footage.ipynb` (~1 h): FP16/TensorRT speed + agreement check; five new clips.
2. `train_people_rfdetr_large.ipynb` (~4-7 h): RF-DETR Large people model.
3. Keypoints model, open licence (Claude prepares the notebook first; see below).

## Tomorrow, you

- Upload new test clips to `MyDrive/Playbook/video_downloads/` (night matches preferred; full
  matches or 30-60 s in-play clips; matches not in the annotation pool are the best test).
- Keypoints: label the new keypoint version (annotation pool) if not done yet; tell Claude the
  Roboflow project/version to train from.

## Tomorrow, Claude

1. Score the new clips (no ground truth: ID fragmentation, impossible jumps, ball found rate) and
   send side-by-side videos (`tools/render_run.py`).
2. Pick `MODEL_ACCEL` from the speed test (only a mode that reproduces the plain model).
3. Compare Large vs Medium people model on both CVAT clips; switch if better.
4. Keypoints: notebook to train an open (Apache 2.0) RF-DETR keypoint model on our keypoint
   dataset; check the licence of the keypoint weights/code before training; evaluate against the
   CVAT pitch-line check (HILAL-AHLI) and the old model, then switch `FIELD_MODEL_*`.
5. **Team classification: it mixes teams (reported by you).** Measure it first: give each player
   track in the two CVAT clips its true team (about 30 tracks per clip, quick to label from crops)
   and score the pipeline's `team_id`. Likely fixes, in order: decide the team once per track by
   majority vote over its whole life (like the role vote), instead of frame by frame; fit the two
   kit colours on many frames of clean, unoccluded torso crops with referees and goalkeepers left
   out; assign goalkeepers by the side of the pitch they defend. Must be solid before a full match,
   because pass tagging and team statistics depend on it.
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
