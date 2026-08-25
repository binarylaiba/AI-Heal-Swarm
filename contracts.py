"""Data contracts and schemas for Autonomous Multi-Agent Software Development Swarm.

Built with Pydantic v2 to enforce strict validation, type safety,
and seamless state transitions across agent pipelines.
"""

import re
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FileArtifact(BaseModel):
    """Represents an individual source, test, or configuration file artifact."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    filename: str = Field(
        ...,
        description="Relative file path (e.g., 'calculator.py', 'test_calculator.py').",
        min_length=1,
    )
    content: str = Field(
        ...,
        description="Raw code or file content strictly without markdown fences.",
        min_length=1,
    )
    file_type: Literal["source", "test", "config"] = Field(
        ...,
        description="Categorization of the file artifact within the project lifecycle.",
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Ensure filename is not empty and has a valid non-whitespace format."""
        clean_val = value.strip()
        if not clean_val:
            raise ValueError("Filename cannot be empty or solely whitespace.")
        return clean_val

    @field_validator("content")
    @classmethod
    def validate_and_sanitize_content(cls, value: str) -> str:
        """Ensure code content is non-empty and strip accidental markdown code fences."""
        clean_val = value.strip()
        if not clean_val:
            raise ValueError("File content cannot be empty or solely whitespace.")

        # Strip enclosing markdown code block fences if present (e.g. ```python ... ``` or ``` ... ```)
        fence_pattern = r"^```[a-zA-Z0-9_\-\+]*\s*\n?(.*?)\n?```$"
        match = re.match(fence_pattern, clean_val, re.DOTALL)
        if match:
            clean_val = match.group(1).strip()

        if not clean_val:
            raise ValueError("File content is empty after stripping markdown fences.")

        return clean_val


class DebuggingPatch(BaseModel):
    """Represents a debugging patch proposed by an agent to fix bugs or failing tests."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    root_cause_analysis: str = Field(
        ...,
        description="Detailed diagnosis and root cause explanation of the issue.",
        min_length=1,
    )
    files_to_update: List[FileArtifact] = Field(
        default_factory=list,
        description="List of file artifacts to be created or overwritten by this patch.",
    )

    @field_validator("root_cause_analysis")
    @classmethod
    def validate_root_cause(cls, value: str) -> str:
        clean_val = value.strip()
        if not clean_val:
            raise ValueError("Root cause analysis must be provided and cannot be empty.")
        return clean_val


class SwarmProject(BaseModel):
    """Represents the complete state of a project managed by the agent swarm."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    project_name: str = Field(
        ...,
        description="Name of the software project.",
        min_length=1,
    )
    architecture_summary: str = Field(
        ...,
        description="Architectural design overview and module specifications.",
        min_length=1,
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of PyPI package dependencies (e.g., ['pydantic>=2.0', 'pytest']).",
    )
    files: List[FileArtifact] = Field(
        default_factory=list,
        description="List of all file artifacts in the project repository.",
    )
    test_command: str = Field(
        default="python -m unittest discover -s . -p 'test*.py'",
        description="Shell command used by validation agents to execute the test suite.",
    )

    @field_validator("project_name", "architecture_summary", "test_command")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        clean_val = value.strip()
        if not clean_val:
            raise ValueError("Field cannot be empty or solely whitespace.")
        return clean_val

    def get_file(self, filename: str) -> Optional[FileArtifact]:
        """Retrieve a file artifact by its filename."""
        normalized = filename.strip()
        for file in self.files:
            if file.filename == normalized:
                return file
        return None

    def update_files(self, new_files: List[FileArtifact]) -> None:
        """Update existing files by filename or append new files if they do not exist.

        Args:
            new_files: List of FileArtifact objects to overwrite or insert into the project.
        """
        if not new_files:
            return

        file_index = {file.filename: i for i, file in enumerate(self.files)}

        for new_file in new_files:
            if new_file.filename in file_index:
                # Overwrite existing file artifact in-place
                idx = file_index[new_file.filename]
                self.files[idx] = new_file
            else:
                # Append new file artifact and register its index
                self.files.append(new_file)
                file_index[new_file.filename] = len(self.files) - 1

    def apply_patch(self, patch: DebuggingPatch) -> None:
        """Convenience method to apply a DebuggingPatch directly to the project files."""
        self.update_files(patch.files_to_update)
