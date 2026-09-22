# Pipeline benchmark

Source: `/content/bench/half1/gt_window.csv` · id column `display_track_id` · frames 1066–3333 · 1.26 min @ 30.00 fps

## Detection

| metric | value |
|---|---|
| goalkeepers_per_frame | mean=0.155, median=0.000, p10=0.000, p90=1.000 |
| players_per_frame | mean=14.510, median=14.000, p10=11.000, p90=18.000 |
| frames_with_zero_players | 0 |
| referees_per_frame | mean=1.061, median=1.000, p10=1.000, p90=2.000 |

## Ball

| metric | value |
|---|---|
| frames_with_ball | 1.000 |
| frames_with_multiple_ball_rows | 0 |
| frames_with_real_ball_detection | 0.622 |
| frames_with_interpolated_ball | 0.378 |
| ball_rows_off_pitch | 0.032 |

## Homography

| metric | value |
|---|---|
| homography_ok_rate | 0.987 |
| homography_state_share | ok=0.984, none=0.013, manual=0.004 |
| kp_used_median | 9.000 |
| frames_kp_used_lt4 | 0 |
| people_rows_with_pitch_xy | 0.987 |

## Identity — goalkeeper (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 1 |
| expected_ids | 2 |
| ids_over_expected | 0.500 |
| new_ids_per_minute | 0.794 |
| segments | 8 |
| segment_seconds_mean | 1.596 |
| segment_seconds_median | 0.417 |
| segment_seconds_p90 | 4.476 |
| row_share_in_ids_lasting_60s | 0.000 |

## Identity — player (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 20 |
| expected_ids | 20 |
| ids_over_expected | 1.000 |
| new_ids_per_minute | 15.874 |
| segments | 244 |
| segment_seconds_mean | 4.566 |
| segment_seconds_median | 0.683 |
| segment_seconds_p90 | 12.696 |
| row_share_in_ids_lasting_60s | 0.465 |

## Identity — referee (`display_track_id`)

| metric | value |
|---|---|
| unique_ids | 2 |
| expected_ids | 1 |
| ids_over_expected | 2.000 |
| new_ids_per_minute | 1.587 |
| segments | 29 |
| segment_seconds_mean | 3.034 |
| segment_seconds_median | 0.300 |
| segment_seconds_p90 | 8.106 |
| row_share_in_ids_lasting_60s | 0.798 |

## Impossible jumps

| metric | value |
|---|---|
| player_minutes_observed | 18.282 |
| image_jumps | 58 |
| pitch_jumps | 8663 |
| image_jumps_per_player_minute | 3.173 |
| pitch_jumps_per_player_minute | 473.865 |
| thresholds | max_speed_mps=12.000, max_body_heights_per_frame=1.000 |

## Duplicates

| metric | value |
|---|---|
| duplicate_id_rows_in_same_frame | 0 |

## Team labels

| metric | value |
|---|---|
| players_with_team_label | 20 |
| ids_with_team_flip | 4 |
| team_flips_total | 13 |
| unknown_team_row_share | 0.000 |
