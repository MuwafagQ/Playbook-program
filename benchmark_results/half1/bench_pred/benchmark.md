# Pipeline benchmark

Source: `/content/bench/half1/run/per_frame_tracks.csv` · id column `display_track_id` · frames 1066–3333 · 1.26 min @ 30.00 fps

## Detection

| metric | value |
|---|---|
| goalkeepers_per_frame | mean=0.154, median=0.000, p10=0.000, p90=1.000 |
| players_per_frame | mean=14.910, median=15.000, p10=11.000, p90=19.000 |
| frames_with_zero_players | 0 |
| referees_per_frame | mean=1.547, median=2.000, p10=1.000, p90=2.000 |

## Ball

| metric | value |
|---|---|
| frames_with_ball | 0.952 |
| frames_with_multiple_ball_rows | 0 |
| frames_with_real_ball_detection | 0.630 |
| frames_with_interpolated_ball | 0.322 |
| ball_rows_off_pitch | 0.042 |

## Homography

| metric | value |
|---|---|
| homography_ok_rate | 0.987 |
| homography_state_share | ok=0.987, none=0.013 |
| kp_used_median | 9.000 |
| frames_kp_used_lt4 | 0 |
| people_rows_with_pitch_xy | 0.987 |

## Identity — goalkeeper (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 2 |
| expected_ids | 2 |
| ids_over_expected | 1.000 |
| new_ids_per_minute | 1.587 |
| segments | 6 |
| segment_seconds_mean | 2.161 |
| segment_seconds_median | 1.233 |
| segment_seconds_p90 | 4.866 |
| row_share_in_ids_lasting_60s | 0.000 |

## Identity — player (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 26 |
| expected_ids | 20 |
| ids_over_expected | 1.300 |
| new_ids_per_minute | 20.636 |
| segments | 375 |
| segment_seconds_mean | 3.029 |
| segment_seconds_median | 0.233 |
| segment_seconds_p90 | 10.699 |
| row_share_in_ids_lasting_60s | 0.229 |

## Identity — referee (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 2 |
| expected_ids | 1 |
| ids_over_expected | 2.000 |
| new_ids_per_minute | 1.587 |
| segments | 24 |
| segment_seconds_mean | 5.191 |
| segment_seconds_median | 0.433 |
| segment_seconds_p90 | 10.559 |
| row_share_in_ids_lasting_60s | 0.543 |

## Identity — goalkeeper (raw BoT-SORT id)

| metric | value |
|---|---|
| unique_ids | 15 |
| expected_ids | 2 |
| ids_over_expected | 7.500 |
| new_ids_per_minute | 11.906 |
| segments | 22 |
| segment_seconds_mean | 0.629 |
| segment_seconds_median | 0.150 |
| segment_seconds_p90 | 1.590 |
| row_share_in_ids_lasting_60s | 0.000 |

## Identity — player (raw BoT-SORT id)

| metric | value |
|---|---|
| unique_ids | 170 |
| expected_ids | 20 |
| ids_over_expected | 8.500 |
| new_ids_per_minute | 134.930 |
| segments | 315 |
| segment_seconds_mean | 3.637 |
| segment_seconds_median | 0.300 |
| segment_seconds_p90 | 11.346 |
| row_share_in_ids_lasting_60s | 0.000 |

## Identity — referee (raw BoT-SORT id)

| metric | value |
|---|---|
| unique_ids | 75 |
| expected_ids | 1 |
| ids_over_expected | 75.000 |
| new_ids_per_minute | 59.528 |
| segments | 183 |
| segment_seconds_mean | 0.769 |
| segment_seconds_median | 0.200 |
| segment_seconds_p90 | 1.933 |
| row_share_in_ids_lasting_60s | 0.000 |

## Impossible jumps

| metric | value |
|---|---|
| player_minutes_observed | 18.785 |
| image_jumps | 320 |
| pitch_jumps | 8758 |
| image_jumps_per_player_minute | 17.035 |
| pitch_jumps_per_player_minute | 466.227 |
| thresholds | max_speed_mps=12.000, max_body_heights_per_frame=1.000 |

## Duplicates

| metric | value |
|---|---|
| duplicate_id_rows_in_same_frame | 326 |

## Team labels

| metric | value |
|---|---|
| players_with_team_label | 26 |
| ids_with_team_flip | 0 |
| team_flips_total | 0 |
| unknown_team_row_share | 0.034 |

## Pipeline KPI (from run)

| metric | value |
|---|---|
| source_video | /content/HIL-HAZ_half1.mp4 |
| source_fps | 30.002 |
| processed_frames | 2268 |
| effective_fps | 1.041 |
| avg_players_per_frame | 14.910 |
| ball_detect_coverage | 0.630 |
| ball_interp_coverage | 0.322 |
| ball_roi_recovery_rate | 0.035 |
| homography_ok_rate | 0.987 |
| homography_available_rate | 0.987 |
| valid_projection_ratio | 0.987 |
| unique_tracks | 32 |
| short_track_ratio | 0.000 |
| team_unknown_ratio | 0.000 |
| display_evictions_player | 0 |
| display_evictions_goalkeeper | 4 |
| display_evictions_referee | 66 |
| stab_player_p1_frame_continuity | 34051 |
| stab_player_p2_raw_continuity | 4106 |
| stab_player_p3_hungarian | 99 |
| stab_player_p4_spatial_rescue | 43 |
| stab_player_new_ids | 26 |
| stab_player_capped_forced_reuse | 222 |
| stab_goalkeeper_p1_frame_continuity | 259 |
| stab_goalkeeper_p2_raw_continuity | 276 |
| stab_goalkeeper_p3_hungarian | 28 |
| stab_goalkeeper_p4_spatial_rescue | 14 |
| stab_goalkeeper_new_ids | 3 |
| stab_goalkeeper_capped_forced_reuse | 32 |
| stab_referee_p1_frame_continuity | 2101 |
| stab_referee_p2_raw_continuity | 592 |
| stab_referee_p3_hungarian | 43 |
| stab_referee_p4_spatial_rescue | 8 |
| stab_referee_new_ids | 3 |
| stab_referee_capped_forced_reuse | 22 |
