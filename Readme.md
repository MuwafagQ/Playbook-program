# Soccer Analytics Pipeline (Detection • Tracking • Teams • Homography)

This project runs a soccer analytics pipeline that:
- Detects players/ball/referees/goalkeepers
- Tracks identities across frames
- Classifies teams (optional)
- Estimates homography from field keypoints and projects entities to pitch coordinates
- Outputs:
  - Annotated video (`annotated.mp4`)
  - Per-frame CSV (`per_frame_tracks.csv`)
  - Basic run logs (via console + CSV metadata columns)

It can be run:
1) As a local script (`main.py`)
2) As an API (FastAPI) with background jobs (`api.py`)
3) With a Streamlit frontend (`streamlit_app.py`)

---

## 1) Project Structure

Typical layout:
your_project/
main.py
api.py
streamlit_app.py
requirements.txt
core/
vision/
geometry/
io_utils/
outputs/
uploads/


---

## 2) Requirements

- Python 3.10+ (Python 3.11 recommended)
- ffmpeg installed (recommended for reliable video IO)

### Install Python dependencies
From the project root:

```bash
pip install -r requirements.txt
If you are adding the API + Streamlit, ensure these are included in requirements.txt:

fastapi

uvicorn[standard]

python-multipart

streamlit

requests

3) Configuration (Roboflow / Model IDs)
This project loads models published on Roboflow. You must provide:

ROBOFLOW_API_KEY

PLAYER_MODEL_ID

FIELD_MODEL_ID

Option A: Environment Variables (recommended)
Set these in your shell:

PowerShell (Windows)

setx ROBOFLOW_API_KEY "YOUR_KEY"
setx PLAYER_MODEL_ID "YOUR_ROBOFLOW_PLAYER_MODEL_ID"
setx FIELD_MODEL_ID "YOUR_ROBOFLOW_FIELD_MODEL_ID"
Restart your terminal after setx.

Mac/Linux

export ROBOFLOW_API_KEY="YOUR_KEY"
export PLAYER_MODEL_ID="YOUR_ROBOFLOW_PLAYER_MODEL_ID"
export FIELD_MODEL_ID="YOUR_ROBOFLOW_FIELD_MODEL_ID"
Option B: .env file
If your load_settings() supports .env, create a .env in project root:

ROBOFLOW_API_KEY=YOUR_KEY
PLAYER_MODEL_ID=YOUR_ROBOFLOW_PLAYER_MODEL_ID
FIELD_MODEL_ID=YOUR_ROBOFLOW_FIELD_MODEL_ID

3b) Firebase Auth (Google + email/password login)
The API (api.py) and the Streamlit app (streamlit_app.py) are protected by Firebase Auth.
Both must be configured before either will work — the Streamlit login screen won't render, and
the API will reject every request, until these are set.

In the Firebase console:
1. Enable the Google and Email/Password sign-in providers under Authentication -> Sign-in method.
2. Project settings -> General -> Your apps -> add a Web app, copy its config values.
3. Project settings -> Service accounts -> Generate new private key, save the JSON file
   somewhere local (do NOT commit it).

Add to your .env:

FIREBASE_SERVICE_ACCOUNT_JSON=/path/to/serviceAccountKey.json
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_API_KEY=your-web-api-key
FIREBASE_AUTH_DOMAIN=your-project-id.firebaseapp.com
FIREBASE_APP_ID=your-web-app-id

Install the added dependency:

pip install firebase-admin

How it works:
- streamlit_app.py embeds the Firebase Web SDK to show a Google/email sign-in screen and
  obtains a Firebase ID token client-side.
- Every request to the API attaches that token as `Authorization: Bearer <token>`.
- api.py verifies the token via firebase-admin (auth.py) on every request, and scopes each
  job to the uid that created it — one signed-in user cannot see another's jobs/artifacts.

Note: signInWithPopup (used for the Google button) runs inside a Streamlit embedded-HTML
iframe; if your browser blocks the popup there, email/password sign-in is unaffected — only
the Google button is at risk of this.

4) Run as a Script (main.py)
Edit the bottom of main.py:

if __name__ == "__main__":
    SOURCE_VIDEO = r"C:\path\to\match.mp4"
    main(SOURCE_VIDEO, out_dir="outputs", enable_team=True)
Run:

python main.py
Outputs:

outputs/annotated.mp4

outputs/per_frame_tracks.csv

Notes
If your machine is slow / hangs, run in “demo mode”:

enable_team=False (team classification is heavy)

or reduce team fitting:

fit_team_stride=60

fit_team_max_frames=20

Example:

main(SOURCE_VIDEO, out_dir="outputs", enable_team=False)
5) Run as an API (FastAPI)
The API accepts a video upload (mp4), creates a job, runs the pipeline in the background, and exposes artifacts.

Start the API server
From project root:

uvicorn api:app --reload --host 0.0.0.0 --port 8000
Endpoints
POST /analyze-video
Upload an MP4 and start processing

Returns job_id immediately

Example (curl):

curl -X POST "http://127.0.0.1:8000/analyze-video?enable_team=false" \
  -F "file=@HILAL-AHLI_match_B_up1.mp4"
Response:

{
  "job_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "queued"
}
GET /jobs/{job_id}
Check status:

curl "http://127.0.0.1:8000/jobs/<job_id>"
Status values:

queued

running

done

failed

GET /jobs/{job_id}/artifacts/{filename}
Download artifacts once done:

annotated.mp4

per_frame_tracks.csv

Example:

curl -O "http://127.0.0.1:8000/jobs/<job_id>/artifacts/annotated.mp4"
curl -O "http://127.0.0.1:8000/jobs/<job_id>/artifacts/per_frame_tracks.csv"
Where files are stored
Uploaded videos: uploads/{job_id}.mp4

Outputs: outputs/{job_id}/

6) Run with Streamlit Frontend
Streamlit provides a UI to upload an MP4, trigger processing, poll status, and download outputs.

Start API first (Terminal 1)
uvicorn api:app --reload --port 8000
Start Streamlit (Terminal 2)
streamlit run streamlit_app.py
Open the Streamlit URL printed in the terminal (usually):

http://localhost:8501

What you can do in the UI
Upload MP4

Toggle “Enable team classification” (recommended OFF for slower laptops)

Click “Analyze video”

Watch status updates

View the annotated output

Download the CSV

6b) Benchmarking (tools/benchmark.py)
The regression test for every later change (homography smoothing, Re-ID work, etc.).

Primary: accuracy on the hand-verified windows (notebooks/benchmark_windows.ipynb, Colab GPU).
Runs the pipeline unattended on half1 frames 1066-3333 and half2 frames 4270-5720 of the
full half videos (main.py --start-frame/--end-frame keeps absolute frame numbers), then
scores it against per_frame_tracks_half*_unified.csv:

   python -m tools.benchmark score-gt --pred run/per_frame_tracks.csv --gt per_frame_tracks_half1_unified.csv

Reports IDF1 (share of player-frames carrying the right identity), ID switches per
player-minute, ids that cover 2+ real players (mid-track swaps), MOTA, and per-player
id accuracy (gt_per_player.csv) with every switch listed (gt_switches.csv). GT rows
fabricated by gap interpolation (notes=idfix_interp) are excluded. It also checks the
frame alignment and box scale, and warns if they look off. Caveat: the truth boxes come
from the same detector, so detection recall/precision here reflect the cleanup, not true
detector misses; the identity numbers are the ones to trust.

Record once, replay many times. The models are the slow part (about 1 frame/s on a T4).
Record their outputs once, then re-run tracking/Re-ID experiments with no models or GPU:

   python main.py --source-video half1.mp4 --start-frame 1066 --end-frame 3333 --out-dir runs/rec --record-cache runs/rec/cache --no-video
   python main.py --replay-cache runs/rec/cache --replay-video proxy_half1.mp4 --out-dir runs/exp1 --no-video --enable-team

--replay-video is a reduced-resolution clip of just the window (frame 0 = start frame;
the notebook makes it); without it the replay reads --source-video. A replay warns if
detection settings differ from the recording. Every run prints and saves per-stage
timing (kpi_summary.json: time_ms_per_frame_*).

Secondary: proxy metrics without ground truth (any video, e.g. a full half later).
Protocol:
1. Baseline run — one full, untouched half (not a pre-trimmed clip), baseline.env settings,
   MAX_FRAMES=0, no manual edits:

   python main.py --source-video half1_full.mp4 --out-dir runs/baseline --enable-team

2. Score it:

   python -m tools.benchmark run --csv runs/baseline/per_frame_tracks.csv --kpi runs/baseline/kpi_summary.json --out-dir runs/baseline/bench

   Writes benchmark.json / benchmark.md, suspects.csv (every impossible jump, frame + id,
   ready to review), and spotcheck.csv (12 random 20 s windows).

3. Human spot-check (the only way to catch silent swaps that don't cause a jump): open
   annotated.mp4 at each spotcheck.csv window, fill players_checked and id_switches_found, then:

   python -m tools.benchmark score-spotcheck --spotcheck runs/baseline/bench/spotcheck.csv --fps 25

   Gives ID switches per player-minute with a 95% confidence interval.

4. Reference point — run step 2 on the hand-cleaned per_frame_tracks_half*_unified.csv files
   (fps 25) to see how far unattended output is from the manually cleaned result.

5. After any pipeline change, re-run steps 1-2 on the same video and diff:

   python -m tools.benchmark compare runs/baseline/bench/benchmark.json runs/candidate/bench/benchmark.json

What the metrics mean:
- identity.*: unique ids vs expected, new ids per minute, mean/median continuous segment
  length — the fragmentation picture. identity_raw_tracker shows the same for raw BoT-SORT
  ids, so the gain from IDStabilizer is visible.
- jumps.image_jumps: the same id moves more than a body-height per frame in the image —
  physically impossible, independent of the homography, so almost always an ID swap.
- jumps.pitch_jumps: implied ground speed over 12 m/s while image motion is normal —
  homography jitter, not identity.
- team.ids_with_team_flip: an id whose team label changes — a swap or a classifier error.
- pipeline_kpi.stab_player_*: how IDStabilizer assigned ids (per matching pass, new ids,
  forced reuses when the id cap is hit); display_evictions_*: on-screen label slots
  handed to a different player.

7) Troubleshooting
A) “System hangs / very slow”
Common causes:

Team classification embeddings (SigLIP) are heavy

CPU-only inference on long videos

Fixes:

Run with enable_team=False

Reduce team fitting load:

increase fit_team_stride (e.g., 60)

reduce fit_team_max_frames (e.g., 20)

Test with a short clip first (30–60s)

