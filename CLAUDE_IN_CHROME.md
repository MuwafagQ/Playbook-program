# Prompts for Claude in Chrome

Copy one block into Claude in Chrome (the browser extension) with the Colab or app tab open. Each
prompt says what to do, what success looks like, and what to report back, so the result can be
pasted straight to Claude Code. Notebook links open the latest version of the branch
`claude/setup-gpu-video-testing-JhgUH`.

Common rules for every notebook run (already inside each prompt):
- Never edit a cell. Never type or paste API keys or tokens.
- If Colab asks to connect to Google Drive, accept for the account that has `MyDrive/Playbook`.
- If the runtime disconnects, reconnect and choose Runtime -> Run all again: finished work is skipped.
- If a cell fails, stop and report the last 30 lines of that cell's output.

---

## 1. Full match (GPU, about 7 hours)

```
Open https://colab.research.google.com/github/MuwafagQ/Playbook-program/blob/claude/setup-gpu-video-testing-JhgUH/notebooks/full_match.ipynb
1. Runtime -> Change runtime type -> L4 GPU -> Save.
2. Runtime -> Run all. Accept the Google Drive connection when asked.
3. Keep this tab open. Every 20 minutes, check the output of Cell 3: each finished segment prints a line
   like "02_1H_05-10: 21 min, 7.3 frames/s, ball 92%, pitch 100% | 18 segments left". If the runtime has
   disconnected, reconnect and Run all again (finished segments are skipped).
4. When the last cell prints "Pass Review data in ...", the run is done.
Never edit cells and never enter any key or token. If a cell shows an error, stop and copy its last 30 lines.
Report: the segment lines from Cell 3, the printed summary from Cell 5 (players joined, team alignment
margins), the "video frames ... OK" line from Cell 6, and the joining numbers from Cell 8.
```

## 2. GSR re-score on SoccerNet (GPU, about 4-5 hours)

```
Open https://colab.research.google.com/github/MuwafagQ/Playbook-program/blob/claude/setup-gpu-video-testing-JhgUH/notebooks/gsr_rescore.ipynb
Runtime -> Change runtime type -> L4 GPU -> Save, then Runtime -> Run all. Accept the Drive connection.
Keep the tab open; every 20 minutes check Cell 3 ("SNGS-xxx: 4.8 min ... | N left"). If disconnected,
reconnect and Run all again. Never edit cells or enter keys. On an error, copy the last 30 lines.
Report: the table printed by the last cell, and how many clips it says were scored.
```

## 3. Review export for a match that was run before the app existed (CPU, 3 minutes)

```
Open https://colab.research.google.com/github/MuwafagQ/Playbook-program/blob/claude/setup-gpu-video-testing-JhgUH/notebooks/review_export.ipynb
Runtime -> Change runtime type -> CPU -> Save, then Runtime -> Run all. Accept the Drive connection.
Never edit cells or enter keys. Report the lines printed by Cell 2 (one per segment) and its last line.
```

## 4. Check the Pass Review app when something looks wrong

```
In the open Pass Review tab (claude.ai artifact "Pass Review · match_video_2"):
1. Note the segment selected, the tally at the top, and any red message under the Save buttons.
2. Open the browser developer console and copy any red error lines.
3. Take a screenshot of the video area with the boxes.
Do not press Save, Not a pass, Delete or Reopen. Report the three items above.
```
