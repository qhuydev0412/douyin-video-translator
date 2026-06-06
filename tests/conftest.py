"""Shared test fixtures and configuration."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_storage(tmp_path: Path) -> Path:
    """Provide a temporary storage directory for tests."""
    storage_dir = tmp_path / "storage" / "jobs"
    storage_dir.mkdir(parents=True)
    return storage_dir
