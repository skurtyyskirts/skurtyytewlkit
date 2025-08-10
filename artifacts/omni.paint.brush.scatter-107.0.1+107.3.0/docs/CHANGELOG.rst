# CHANGELOG

This document records all notable changes to ``omni.paint.brush.scatter`` extension.
This project adheres to `Semantic Versioning <https://semver.org/>`_.

# [107.0.1] - 2025-03-12
### Changed
- Enabled Arm Builds

# [107.0.0] - 2025-02-11
### Changed
- remove dependence of omni.physx.zerogravity, not support 'physics' and 'gravity'

# [105.1.7] - 2023-12-06
### Fixed
- use async thumbnial load

# [105.1.6] - 2023-08-25
### Fixed
- catch a exception when open layer

# [105.1.5] - 2023-08-16
### Add
- add tooltip for 'Add Asset' button

# [105.1.4] - 2023-06-26
### Fxied
- jump version to 105.1.4 avoid confuse
- disable test on vp1.0 and update golden image

# [105.1.1] - 2023-04-23
### Fxied
- Fixed unit test on latest Kit-SDK

# [105.1.0] - 2023-01-03
### Changed
- remove dependence of omni.kit.widgets.custom

# [105.0.2] - 2022-12-16
### Fixed
- Add asset not work with new asset browsers

# [105.0.1] - 2022-12-15
### Fixed
- cherry pick "fixed messages when painting without asset"

# [105.0.0] - 2022-11-21
### Changed
- cherry pick from 104.1 (erase range)

# [104.2.9] - 2022-10-25
### Changed
- Updated asset paths.

# [104.2.8] - 2022-10-21
### Changed
- Fixed an issue that caused the property window to flicker after each stroke.

# [104.2.7] - 2022-09-30
### Changed
- Use Actions.

# [104.2.6] - 2022-08-08
### Fixed
- Fixed tests

# [104.2.5] - 2022-07-22
### Changed
- bypass image compare in test

# [104.2.4] - 2022-07-18
### Changed
- add Xform on paint root prim

# [104.2.3] - 2022-07-13
### Fixed
- Workaround when no ids in pointinstancer (OM-56195)

# [104.2.2] - 2022-06-28
### Fixed
- Fixed version update

# [104.2.1] - 2022-06-17
### Changed
- Update for new physx package. ZeroG is not its own extension.

# [104.2.0] - 2022-06-14
### Changed
- updated to viewport 2.0

# [104.1.6] - 2022-04-15
### Changed
- merge from release 103

# [104.1.5] - 2022-03-24
### Changed
- Changed some warning to info message

# [104.1.4] - 2022-03-21
### Added
- Support add asset from Asset Store

# [104.1.3] - 2022-03-16
### Changed
- Set default `lock_selection` to False.

# [104.1.2] - 2022-03-14
### Changed
- updated standard parameter functions

# [104.1.1] - 2022-03-10
### Changed
- set parent of painted assets as Xform
- update tests

# [104.1.0] - 2022-02-18
### Changed
- rename
- update to new paint system

# [104.0.0] - 2022-01-27
### Changed
- Added writeTarget to extension.toml file

# [103.1.0] - 2021-12-06
### Changed
- Changed `omni.kit.viewport` to `omni.kit.viewport_legacy` due to kit-sdk API change.

# [103.0.4] - 2021-10-21
### Fixed
- Fixed penetration issue when flooding with physics on.
### Added
- Support add asset from Asset Library

# [103.0.3] - 2021-09-24
### Fixed
- Bugfixes.

# [103.0.2] - 2021-08-19
### Changed
- Moved content to S3 server.

# [103.0.1] - 2021-08-06
### Changed
- Bugfixes.

#### [102.1.2] - 2021-06-21
### Fixed
- Version changes.

#### [102.1.1] - 2021-06-21
### Fixed
- Version changes.

#### [102.1.0] - 2021-05-25
### Fixed
- Version changes.

#### [0.14.1] - 2021-05-21
### Fixed
- Naming changes.

#### [0.14.0] - 2021-05-13
### Fixed
- Bug fixes.

#### [0.13.1] - 2021-05-11
### Fixed
- Sampling for stage local prototype prims.

#### [0.13.0] - 2021-05-04
### Changed
- Normalized stamping spacing to brush size.

#### [0.12.0] - 2021-05-04
### Changed
- Prepare for 101 release.

#### [0.11.0] - 2021-05-01
### Changed
- improved flood paint
- Assets field resizeable

#### [0.10.1] - 2021-04-29
### Changed
- Make plugin compatible with both Kit SDK 101 and 102.

## [0.10.0] - 2021-04-29
### Changed
- `Assets` and `Brush Parameters` section now supports undo/redo.
- B + mouse wheel scroll to change brush size now support undo/redo.

## [0.9.0] - 2021-04-20
### Changed
- Standardized brush settings styling.

## [0.8.0] - 2021-04-19
### Changed
- Undo per stroke, not per stamp
- Improve flooding erase

## [0.7.0] - 2021-04-12
### Fixed
- dependence versions define at multiply places

## [0.6.0] - 2021-04-10
### Fixed
- Wrong paint mode buttons status.
- Save relative asset file path in brush define file.

## [0.5.11] - 2021-04-07
### Changed
- Update paint tool dependency version.

## [0.5.10] - 2021-04-06
### Changed
- Use event stream to control UI state.

## [0.5.9] - 2021-04-01
### Changed
- support drag and drop prim on asset UI
- un-registers brush when extensions unload

## [0.5.8] - 2021-03-29
### Changed
- update for omni.paint.system@0.11.0

## [0.5.7] - 2021-03-24
### Changed
- bind omni.kit.widgets.custom and omni.paint.system version

## [0.5.6] - 2021-03-23
### Changed
- expand brush parameters panel by default
- set selected on last strok painted assets
- turn painted asset up-axis by current stage up-axis

## [0.5.5] - 2021-03-11
### Changed
- change message box type to omni.kit.notification_manager

## [0.5.4] - 2021-02-26
### Added
- add warning dialog when the painting stamp has large number of assets
### changed
- update to omni.paint.system@0.9.0

## [0.5.3] - 2021-01-26
### Changed
- Remove omni.kit.editor

## [0.5.2] - 2021-01-25
### Changed
- In kit, using "/PaintTool" as root prim path instead of "/View/Tools/Paint"

## [0.5.1] - 2021-1-18
### Fixed
- Asset picker dialog mode

## [0.5.0] - 2020-12-18
### Added
- Add default brush files
- Add pick asset for brush using viewport selection
### Improved
- Make scale distribution image correct when startup and window width Changed
- Update to new file picker
- improve erase performance

## [0.4.6] - 2020-12-08
### Changed
- Update to new file picker

## [0.4.5] - 2020-11-27
### Changed
- Update controls to match omni.kit.widgets.custom

## [0.4.4] - 2020-11-24
### Changed
- Update controls to match omni.kit.widgets.custom

## [0.4.3] - 2020-11-02
### Fixed
- UI layout in kit v100.1.10219

## [0.4.2] - 2020-10-29
- Change structure (remote source/target folder) to enable license packaging

## [0.4.0] - 2020-10-24
### Added
- Add hit mesh when set_data
### Changed
- Change asset indexes to object ids in get_data/set_data
- Asset is part of scatter brush but not paint tool anymore
- Return True in begin_stroke if brush is valid, otherwise False
- Update UI style in kit


## [0.3.0] - 2020-10-20
### Changed
- Refactor interface


## [0.2.1] - 2020-10-11
### Added
- Interface to refresh brush parameters UI from paint tool window


## [0.2.0] - 2020-09-27
### Added
- Erase
- Redo/Undo


## [0.1.0] - 2020-08-02
### Added
- Initial scatter brush implementation
