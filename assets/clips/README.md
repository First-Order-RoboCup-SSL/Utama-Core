# Demo clips

Short highlight clips rendered from rsim replays with
`utama_core/analysis/render_clip.py`. Record the command for every clip added here, so it can
be regenerated when the renderer or the dashboard style changes.

| Clip | Match | Window | Camera |
|---|---|---|---|
| `buildup_pass.mp4` | `split_shape` (yellow) vs `counter_flow`, already 1-0 | t=97–105.5s: interception near midfield, ~5m pass into the attacking third | follow |
| `buildup_pass_full_pitch.mp4` | same | same | full |

```bash
pixi run python -m utama_core.analysis.render_clip --replay replays/scan_split_v_counterflow.npz \
    --t-start 97 --t-end 105.5 --out assets/clips/buildup_pass.mp4
pixi run python -m utama_core.analysis.render_clip --replay replays/scan_split_v_counterflow.npz \
    --t-start 97 --t-end 105.5 --out assets/clips/buildup_pass_full_pitch.mp4 --camera full
```

`replays/` is gitignored, so the source replay only exists on the machine that recorded it.
The window was picked by eye; there is no automated "find a good clip" tool.
