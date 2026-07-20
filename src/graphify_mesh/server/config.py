"""Resolved runtime configuration for one graphify-mesh server process.

Mirrors `graphify_mesh.sync.config.Settings`'s "all paths configurable"
convention so tests never touch a real filesystem tree; the default below is
a placeholder — override via GRAPHIFY_MESH_ROOT for your environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerConfig:
    mesh_root: Path
    registry_path: Path

    @property
    def global_dir(self) -> Path:
        return self.mesh_root / "graphify" / "global"

    @property
    def current_symlink(self) -> Path:
        return self.global_dir / "current"

    @property
    def embeddings_current_symlink(self) -> Path:
        return self.global_dir / "embeddings" / "current"

    @classmethod
    def from_env(
        cls, mesh_root: Path | None = None, registry_path: Path | None = None
    ) -> ServerConfig:
        # Defaults to the current working directory (no machine-specific
        # path); set GRAPHIFY_MESH_ROOT for a real deployment.
        resolved_mesh_root = Path(
            mesh_root or os.environ.get("GRAPHIFY_MESH_ROOT") or Path.cwd()
        ).resolve()
        resolved_registry = Path(
            registry_path
            or os.environ.get(
                "GRAPHIFY_MESH_REGISTRY", str(resolved_mesh_root / "bin" / "registry.json")
            )
        ).resolve()
        return cls(mesh_root=resolved_mesh_root, registry_path=resolved_registry)
