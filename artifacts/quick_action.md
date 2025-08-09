Quick action: prioritized integration recommendations

1) Minimal reuse (fastest path)
- Use Apache-2.0 `kit-extension-sample-scatter` as reference only.
- Implement a new `extensions/omniscatter` with brush UI and PointInstancer painting.
- Author instances under `/World/ScatterAnchors` with batched writes and undo.

2) Moderate reuse (balanced)
- Fork `kit-extension-sample-scatter` and transplant its extension skeleton, command patterns, and PointInstancer authoring utilities.
- Replace world-space randomization with brush stroke sampling + surface raycasts.
- Add prototype picker and thumbnail grid from our ingest index.

3) Maximal reuse (slowest but least risk)
- Vendor the sample as a submodule, refactor into modules consumed by `omniscatter` (UI components, instancer utils).
- Keep license headers; expose a surface-paint mode alongside original random-scatter mode.

Notes
- Across all options, adhere to Apache-2.0 attribution and avoid copying Paint Tool implementation.

