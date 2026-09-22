# Pipeline benchmark

Source: `/content/bench/half2/gt_window.csv` · id column `display_track_id` · frames 4270–5720 · 0.81 min @ 30.00 fps

## Detection

| metric | value |
|---|---|
| goalkeepers_per_frame | mean=0.086, median=0.000, p10=0.000, p90=0.000 |
| players_per_frame | mean=12.236, median=13.000, p10=8.000, p90=16.000 |
| frames_with_zero_players | 0 |
| referees_per_frame | mean=0.982, median=1.000, p10=0.000, p90=2.000 |

## Ball

| metric | value |
|---|---|
| frames_with_ball | 1.000 |
| frames_with_multiple_ball_rows | 0 |
| frames_with_real_ball_detection | 0.548 |
| frames_with_interpolated_ball | 0.452 |
| ball_rows_off_pitch | 0.091 |

## Homography

| metric | value |
|---|---|
| homography_ok_rate | 0.989 |
| homography_state_share | ok=0.983, none=0.011, manual=0.006 |
| kp_used_median | 6.000 |
| frames_kp_used_lt4 | 0 |
| people_rows_with_pitch_xy | 0.988 |

## Identity — goalkeeper (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 1 |
| expected_ids | 2 |
| ids_over_expected | 0.500 |
| new_ids_per_minute | 1.240 |
| segments | 5 |
| segment_seconds_mean | 0.847 |
| segment_seconds_median | 0.400 |
| segment_seconds_p90 | 1.833 |
| row_share_in_ids_lasting_60s | 0.000 |

## Identity — player (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 20 |
| expected_ids | 20 |
| ids_over_expected | 1.000 |
| new_ids_per_minute | 24.811 |
| segments | 90 |
| segment_seconds_mean | 6.615 |
| segment_seconds_median | 4.167 |
| segment_seconds_p90 | 17.567 |
| row_share_in_ids_lasting_60s | 0.000 |

## Identity — referee (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 5 |
| expected_ids | 1 |
| ids_over_expected | 5.000 |
| new_ids_per_minute | 6.203 |
| segments | 18 |
| segment_seconds_mean | 2.698 |
| segment_seconds_median | 1.967 |
| segment_seconds_p90 | 6.710 |
| row_share_in_ids_lasting_60s | 0.000 |

## Impossible jumps

| metric | value |
|---|---|
| player_minutes_observed | 9.864 |
| image_jumps | 27 |
| pitch_jumps | 4956 |
| image_jumps_per_player_minute | 2.737 |
| pitch_jumps_per_player_minute | 502.439 |
| thresholds | max_speed_mps=12.000, max_body_heights_per_frame=1.000 |

## Duplicates

| metric | value |
|---|---|
| duplicate_id_rows_in_same_frame | 0 |

## Team labels

| metric | value |
|---|---|
| players_with_team_label | 20 |
| ids_with_team_flip | 0 |
| team_flips_total | 0 |
| unknown_team_row_share | 0.000 |
