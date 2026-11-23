# Changelog

## [1.0.2]
### Fixed
- Disabled Opacity Micromaps and volumetric atmosphere and pinned the ReSTIR GI indirect mode in the preload `dxvk.conf` to avoid GPU page faults and device loss on RTX 50-series GPUs.

## [1.0.1]
### Changed
- Updated extension metadata for Kit SDK 110 compatibility.

## [1.0.0]

### Added

- Added default-on HdRemix Nsight Aftermath DLL preloading for RTX Remix Toolkit.
- Added separate settings for Kit-owned Aftermath and DXVK/HdRemix Aftermath.
- Added `DXVK_CONFIG_FILE` setup with launch working directory and extension fallback config support.
