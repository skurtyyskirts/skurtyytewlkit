# Changelog
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0]
### Added
- Created the Scatter window: paint/erase/flood controls, brush, placement, scale and target settings, the scatterable asset list with ingestion validation, an Assign Textures toggle with a per-asset Maps column and texture-set override, and brush presets
- Added a Scale Mode control to the SCALE section: Absolute Size gives each placement a height in stage units and shows a Size range, while Multiplier keeps the scale-factor range and its per-axis rows. The asset list reports each asset's measured height in the tooltip under its path
- Added a Prim Budget control to the TARGET section, which sets when the viewport's live stroke counter turns amber and red. The flood estimate no longer reports the flood cap as a prim total in Hit Surface mode, where the prototypes a flood lands on are only known once it runs: it reads `per instance` there and gives the multiplied total only for an anchor whose instances can be counted
- Added an Edit Selection Live toggle at the top of the PLACEMENT section, on by default, with a readout of what a settings change would update. With scatter selected, changing a placement, scale or seed setting updates that scatter as the control is dragged and commits as a single undo step, so a patch can be tuned after it was painted. Nothing outside the selection is touched, which is what keeps two patches painted with two presets on one mesh independent. The readout also reports the placements an edit left alone and why
