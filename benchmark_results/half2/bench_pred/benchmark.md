# Pipeline benchmark

Source: `/content/bench/half2/run/per_frame_tracks.csv` · id column `display_track_id` · frames 4270–5720 · 0.81 min @ 30.00 fps

## Detection

| metric | value |
|---|---|
| goalkeepers_per_frame | mean=0.050, median=0.000, p10=0.000, p90=0.000 |
| players_per_frame | mean=13.558, median=14.000, p10=8.000, p90=17.000 |
| frames_with_zero_players | 0 |
| referees_per_frame | mean=1.758, median=2.000, p10=1.000, p90=2.000 |

## Ball

| metric | value |
|---|---|
| frames_with_ball | 0.928 |
| frames_with_multiple_ball_rows | 0 |
| frames_with_real_ball_detection | 0.553 |
| frames_with_interpolated_ball | 0.375 |
| ball_rows_off_pitch | 0.103 |

## Homography

| metric | value |
|---|---|
| homography_ok_rate | 0.989 |
| homography_state_share | ok=0.989, none=0.011 |
| kp_used_median | 6.000 |
| frames_kp_used_lt4 | 0 |
| people_rows_with_pitch_xy | 0.988 |

## Identity — goalkeeper (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 2 |
| expected_ids | 2 |
| ids_over_expected | 1.000 |
| new_ids_per_minute | 2.481 |
| segments | 5 |
| segment_seconds_mean | 0.487 |
| segment_seconds_median | 0.067 |
| segment_seconds_p90 | 1.267 |
| row_share_in_ids_lasting_60s | 0.000 |

## Identity — player (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 26 |
| expected_ids | 20 |
| ids_over_expected | 1.300 |
| new_ids_per_minute | 32.254 |
| segments | 205 |
| segment_seconds_mean | 3.183 |
| segment_seconds_median | 0.267 |
| segment_seconds_p90 | 9.053 |
| row_share_in_ids_lasting_60s | 0.000 |

## Identity — referee (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 2 |
| expected_ids | 1 |
| ids_over_expected | 2.000 |
| new_ids_per_minute | 2.481 |
| segments | 10 |
| segment_seconds_mean | 8.847 |
| segment_seconds_median | 3.783 |
| segment_seconds_p90 | 23.133 |
| row_share_in_ids_lasting_60s | 0.000 |

## Identity — goalkeeper (raw BoT-SORT id)

| metric | value |
|---|---|
| unique_ids | 6 |
| expected_ids | 2 |
| ids_over_expected | 3.000 |
| new_ids_per_minute | 7.443 |
| segments | 6 |
| segment_seconds_mean | 0.406 |
| segment_seconds_median | 0.050 |
| segment_seconds_p90 | 1.133 |
| row_share_in_ids_lasting_60s | 0.000 |

## Identity — player (raw BoT-SORT id)

| metric | value |
|---|---|
| unique_ids | 106 |
| expected_ids | 20 |
| ids_over_expected | 5.300 |
| new_ids_per_minute | 131.495 |
| segments | 199 |
| segment_seconds_mean | 3.376 |
| segment_seconds_median | 0.233 |
| segment_seconds_p90 | 9.073 |
| row_share_in_ids_lasting_60s | 0.000 |

## Identity — referee (raw BoT-SORT id)

| metric | value |
|---|---|
| unique_ids | 56 |
| expected_ids | 1 |
| ids_over_expected | 56.000 |
| new_ids_per_minute | 69.469 |
| segments | 133 |
| segment_seconds_mean | 0.755 |
| segment_seconds_median | 0.200 |
| segment_seconds_p90 | 1.680 |
| row_share_in_ids_lasting_60s | 0.000 |

## Impossible jumps

| metric | value |
|---|---|
| player_minutes_observed | 10.929 |
| image_jumps | 520 |
| pitch_jumps | 5231 |
| image_jumps_per_player_minute | 47.578 |
| pitch_jumps_per_player_minute | 478.615 |
| thresholds | max_speed_mps=12.000, max_body_heights_per_frame=1.000 |

## Duplicates

| metric | value |
|---|---|
| duplicate_id_rows_in_same_frame | 513 |

## Team labels

| metric | value |
|---|---|
| players_with_team_label | 25 |
| ids_with_team_flip | 0 |
| team_flips_total | 0 |
| unknown_team_row_share | 0.048 |

## Pipeline KPI (from run)

| metric | value |
|---|---|
| source_video | /content/HIL-HAZ_half2.mp4 |
| source_fps | 30.000 |
| processed_frames | 1451 |
| effective_fps | 0.986 |
| avg_players_per_frame | 13.558 |
| ball_detect_coverage | 0.553 |
| ball_interp_coverage | 0.375 |
| ball_roi_recovery_rate | 0.045 |
| homography_ok_rate | 0.989 |
| homography_available_rate | 0.989 |
| valid_projection_ratio | 0.988 |
| unique_tracks | 33 |
| short_track_ratio | 0.061 |
| team_unknown_ratio | 0.038 |
| display_evictions_player | 0 |
| display_evictions_goalkeeper | 3 |
| display_evictions_referee | 98 |
| stab_player_p1_frame_continuity | 21455 |
| stab_player_p2_raw_continuity | 1827 |
| stab_player_p3_hungarian | 59 |
| stab_player_p4_spatial_rescue | 24 |
| stab_player_new_ids | 26 |
| stab_player_capped_forced_reuse | 107 |
| stab_goalkeeper_p1_frame_continuity | 256 |
| stab_goalkeeper_p2_raw_continuity | 157 |
| stab_goalkeeper_p3_hungarian | 24 |
| stab_goalkeeper_p4_spatial_rescue | 8 |
| stab_goalkeeper_new_ids | 3 |
| stab_goalkeeper_capped_forced_reuse | 18 |
| stab_referee_p1_frame_continuity | 1537 |
| stab_referee_p2_raw_continuity | 276 |
| stab_referee_p3_hungarian | 29 |
| stab_referee_p4_spatial_rescue | 2 |
| stab_referee_new_ids | 4 |
| stab_referee_capped_forced_reuse | 24 |
