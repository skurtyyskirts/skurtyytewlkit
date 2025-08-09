How to run (research + Task 2 ingestion)

Prerequisites
- Windows or Linux with Git and Python 3.10+.
- Pillow for image work: `pip install Pillow`.

Steps
1) Open `artifacts/research_summary.md` and `artifacts/compatibility_report.md` to review links and decisions.
2) Manually verify the three primary links listed in the verification checklist below.
3) If adopting code, clone the official sample:
   - `git clone https://github.com/NVIDIA-Omniverse/kit-extension-sample-scatter`
   - Review its `exts/omni.example.ui_scatter_tool` folder and `extension.toml`.

Run Task 2 prototype ingestion
- Example:
  - `python tools/custom/prototype_ingest.py --ingest-dir assets --out-dir extensions/omniscatter/data/prototypes --apply-to-builds`
- Expected outputs:
  - USD prototypes and thumbnails under `extensions/omniscatter/data/prototypes/`.
  - `prototypes_index.json` written there.
  - If `_build/release` or `_build/debug` exist, files are replicated into `.../extensions/omniscatter/data/prototypes/` in each build.

Verification checklist
- Verify these primary links:
  1) Build a Scatter Tool — Omniverse Workflows: https://docs.omniverse.nvidia.com/workflows/latest/extensions/scatter_tool.html
  2) kit-extension-sample-scatter (Apache-2.0): https://github.com/NVIDIA-Omniverse/kit-extension-sample-scatter
  3) Paint Tool extension docs: https://docs.omniverse.nvidia.com/extensions/latest/ext_paint-tool.html


