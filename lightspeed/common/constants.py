from __future__ import annotations

# Name prefix used for meshes
MESH_NAME_PREFIX = "mesh_"

# Regex patterns expected by mesh_anchor_resolver; group(3) captures the 16-char hash
REGEX_MESH_PATH = r"^(.*)(mesh_)([A-Z0-9]{16})(?:/.*)?$"
REGEX_IN_MESH_PATH = r"^(.*)(mesh_)([A-Z0-9]{16})(?:/.*)?$"
REGEX_INSTANCE_PATH = r"^(.*)(inst_)([A-Z0-9]{16})(?:/.*)?$"
REGEX_IN_INSTANCE_PATH = r"^(.*)(inst_)([A-Z0-9]{16})(?:/.*)?$"