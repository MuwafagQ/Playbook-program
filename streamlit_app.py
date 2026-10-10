import time
import requests
import streamlit as st
import streamlit.components.v1 as components

from core.config import load_settings

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Soccer Analytics Demo", layout="wide")


# ---------------------------------------------------------------------------
# Firebase Auth (Google + email/password)
# ---------------------------------------------------------------------------
# Streamlit has no client-side JS runtime of its own, so sign-in is done via
# the Firebase Web SDK loaded inside an embedded HTML component. On success
# the component redirects the *top* window (the actual Streamlit page, not
# the iframe) with the ID token in a query param, which Python then reads
# via st.query_params and stores in session_state for the rest of the session.

def _firebase_config():
    s = load_settings()
    missing = [
        name for name, val in [
            ("FIREBASE_API_KEY", s.FIREBASE_API_KEY),
            ("FIREBASE_AUTH_DOMAIN", s.FIREBASE_AUTH_DOMAIN),
            ("FIREBASE_PROJECT_ID", s.FIREBASE_PROJECT_ID),
            ("FIREBASE_APP_ID", s.FIREBASE_APP_ID),
        ]
        if not val
    ]
    if missing:
        st.error(
            "Firebase web config is not set: " + ", ".join(missing) + ". "
            "Set these in your .env (from Firebase console -> Project settings -> "
            "General -> Your apps -> Web app)."
        )
        st.stop()
    return {
        "apiKey": s.FIREBASE_API_KEY,
        "authDomain": s.FIREBASE_AUTH_DOMAIN,
        "projectId": s.FIREBASE_PROJECT_ID,
        "appId": s.FIREBASE_APP_ID,
    }


def _render_login():
    cfg = _firebase_config()
    components.html(
        f"""
        <div id="fb-auth-root" style="font-family: sans-serif; max-width: 360px;">
          <button id="google-btn" style="width:100%;padding:10px;margin-bottom:12px;
              cursor:pointer;border-radius:6px;border:1px solid #ccc;background:#fff;">
            Sign in with Google
          </button>
          <hr style="margin:16px 0;" />
          <input id="email" type="email" placeholder="Email" style="width:100%;padding:8px;margin-bottom:8px;" />
          <input id="password" type="password" placeholder="Password" style="width:100%;padding:8px;margin-bottom:8px;" />
          <div style="display:flex;gap:8px;">
            <button id="signin-btn" style="flex:1;padding:10px;cursor:pointer;">Sign in</button>
            <button id="signup-btn" style="flex:1;padding:10px;cursor:pointer;">Create account</button>
          </div>
          <p id="fb-error" style="color:#c0392b;margin-top:10px;"></p>
        </div>

        <script type="module">
          import {{ initializeApp }} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
          import {{
            getAuth, GoogleAuthProvider, signInWithPopup,
            signInWithEmailAndPassword, createUserWithEmailAndPassword,
          }} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

          const app = initializeApp({cfg!r});
          const auth = getAuth(app);
          const errorEl = document.getElementById("fb-error");

          function goWithToken(idToken) {{
            const url = new URL(window.top.location.href);
            url.searchParams.set("token", idToken);
            window.top.location.href = url.toString();
          }}

          function showError(e) {{
            errorEl.textContent = e.message || String(e);
          }}

          document.getElementById("google-btn").addEventListener("click", async () => {{
            try {{
              const cred = await signInWithPopup(auth, new GoogleAuthProvider());
              goWithToken(await cred.user.getIdToken());
            }} catch (e) {{ showError(e); }}
          }});

          document.getElementById("signin-btn").addEventListener("click", async () => {{
            const email = document.getElementById("email").value;
            const password = document.getElementById("password").value;
            try {{
              const cred = await signInWithEmailAndPassword(auth, email, password);
              goWithToken(await cred.user.getIdToken());
            }} catch (e) {{ showError(e); }}
          }});

          document.getElementById("signup-btn").addEventListener("click", async () => {{
            const email = document.getElementById("email").value;
            const password = document.getElementById("password").value;
            try {{
              const cred = await createUserWithEmailAndPassword(auth, email, password);
              goWithToken(await cred.user.getIdToken());
            }} catch (e) {{ showError(e); }}
          }});
        </script>
        """,
        height=280,
    )


def _decode_email_unverified(id_token: str) -> str:
    """Best-effort email extraction for display only. NOT used for authorization
    (the backend independently verifies the token signature on every request)."""
    import base64
    import json as _json

    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = _json.loads(base64.urlsafe_b64decode(payload))
        return claims.get("email", "signed-in user")
    except Exception:
        return "signed-in user"


def require_login() -> str:
    """Blocks the rest of the page until the user is signed in. Returns the
    Firebase ID token to attach as a Bearer header on API calls."""
    token_from_url = st.query_params.get("token")
    if token_from_url and not st.session_state.get("id_token"):
        st.session_state["id_token"] = token_from_url
        st.query_params.clear()
        st.rerun()

    if not st.session_state.get("id_token"):
        st.title("⚽ Soccer Analytics Pipeline")
        st.subheader("Sign in to continue")
        _render_login()
        st.stop()

    return st.session_state["id_token"]


id_token = require_login()
auth_header = {"Authorization": f"Bearer {id_token}"}

with st.sidebar:
    st.caption(f"Signed in as **{_decode_email_unverified(id_token)}**")
    if st.button("Sign out"):
        del st.session_state["id_token"]
        st.rerun()

st.title("⚽ Soccer Analytics Pipeline")

st.sidebar.header("Settings")
enable_team = st.sidebar.checkbox("Enable team classification", value=False)
poll_every = st.sidebar.slider("Poll interval (seconds)", 1, 10, 2)

uploaded = st.file_uploader("Upload an MP4 video", type=["mp4"])

if uploaded is not None:
    st.info(f"Selected file: {uploaded.name} ({uploaded.size/1e6:.2f} MB)")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.video(uploaded)

    if st.button("Analyze video"):
        with st.spinner("Uploading to API and starting job..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "video/mp4")}
            resp = requests.post(
                f"{API_BASE}/analyze-video",
                params={"enable_team": str(enable_team).lower()},
                files=files,
                headers=auth_header,
                timeout=120,
            )

        if resp.status_code == 401:
            st.error("Your session expired. Please sign in again.")
            del st.session_state["id_token"]
            st.stop()
        if resp.status_code != 200:
            st.error(f"API error: {resp.status_code} — {resp.text}")
            st.stop()

        data = resp.json()
        job_id = data["job_id"]
        st.success(f"Job created: {job_id}")

        status_box = st.empty()
        progress = st.progress(0)

        # Poll job status
        while True:
            r = requests.get(f"{API_BASE}/jobs/{job_id}", headers=auth_header, timeout=30)
            if r.status_code != 200:
                st.error(f"Failed to fetch job status: {r.status_code} — {r.text}")
                break

            job = r.json()
            status = job["status"]

            status_box.write(f"**Status:** `{status}`")

            if status == "queued":
                progress.progress(10)
            elif status == "running":
                progress.progress(60)
            elif status == "done":
                progress.progress(100)
                st.success("✅ Done!")

                artifacts = job.get("artifacts") or {}
                video_path = artifacts.get("annotated_video")
                csv_path = artifacts.get("csv")

                if video_path:
                    st.subheader("Annotated output")
                    video_bytes = requests.get(f"{API_BASE}{video_path}", headers=auth_header, timeout=60).content
                    st.video(video_bytes)

                if csv_path:
                    st.subheader("Per-frame CSV")
                    csv_bytes = requests.get(f"{API_BASE}{csv_path}", headers=auth_header, timeout=60).content
                    st.download_button(
                        label="Download CSV",
                        data=csv_bytes,
                        file_name="per_frame_tracks.csv",
                        mime="text/csv",
                    )

                break

            elif status == "failed":
                st.error(f"❌ Failed: {job.get('error')}")
                break

            time.sleep(poll_every)
