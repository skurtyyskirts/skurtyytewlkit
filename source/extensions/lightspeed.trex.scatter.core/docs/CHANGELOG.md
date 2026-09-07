# Changelog
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0]
### Added
- Created the scatter brush core: brush settings and presets, seeded disk sampling with falloff and padding, a numpy mesh-surface cache for CPU projection and normals, HdRemix and CPU surface pickers, Remix prototype/instance target resolution, undoable stroke and flood commands, the brush controller, and automatic assignment of each asset's ingested texture set (albedo, normal with its encoding, roughness, height, metallic and emissive) to the placements it paints. Dragging one end of a low/high setting pair past the other carries the other end along instead of refusing the edit
