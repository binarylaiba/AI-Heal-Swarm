"""Unit tests for contracts.py using unittest and Pydantic v2."""

import unittest
from pydantic import ValidationError
from contracts import FileArtifact, SwarmProject, DebuggingPatch


class TestFileArtifact(unittest.TestCase):
    """Test suite for FileArtifact data model."""

    def test_valid_artifact_creation(self):
        artifact = FileArtifact(
            filename="calculator.py",
            content="def add(a, b):\n    return a + b",
            file_type="source",
        )
        self.assertEqual(artifact.filename, "calculator.py")
        self.assertEqual(artifact.file_type, "source")
        self.assertIn("def add", artifact.content)

    def test_strip_markdown_code_fences(self):
        raw_llm_code = "```python\ndef multiply(a, b):\n    return a * b\n```"
        artifact = FileArtifact(
            filename="math_ops.py",
            content=raw_llm_code,
            file_type="source",
        )
        self.assertEqual(artifact.content, "def multiply(a, b):\n    return a * b")

    def test_empty_content_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            FileArtifact(filename="empty.py", content="   ", file_type="source")

    def test_empty_filename_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            FileArtifact(filename="   ", content="x = 1", file_type="source")

    def test_invalid_file_type_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            FileArtifact(
                filename="data.json",
                content="{}",
                file_type="documentation",  # Only 'source', 'test', 'config' allowed
            )


class TestSwarmProject(unittest.TestCase):
    """Test suite for SwarmProject data model and update operations."""

    def setUp(self):
        self.initial_file = FileArtifact(
            filename="main.py",
            content="print('Hello')",
            file_type="source",
        )
        self.project = SwarmProject(
            project_name="AutoSwarmDemo",
            architecture_summary="A modular CLI application.",
            dependencies=["pytest>=7.0.0"],
            files=[self.initial_file],
        )

    def test_default_test_command(self):
        self.assertEqual(
            self.project.test_command,
            "python -m unittest discover -s . -p 'test*.py'",
        )

    def test_get_file(self):
        found = self.project.get_file("main.py")
        self.assertIsNotNone(found)
        self.assertEqual(found.content, "print('Hello')")

        not_found = self.project.get_file("nonexistent.py")
        self.assertIsNone(not_found)

    def test_update_files_overwrite_existing(self):
        updated_file = FileArtifact(
            filename="main.py",
            content="print('Hello, Swarm!')",
            file_type="source",
        )
        self.project.update_files([updated_file])

        self.assertEqual(len(self.project.files), 1)
        self.assertEqual(self.project.files[0].content, "print('Hello, Swarm!')")

    def test_update_files_append_new(self):
        test_file = FileArtifact(
            filename="test_main.py",
            content="def test_main(): pass",
            file_type="test",
        )
        self.project.update_files([test_file])

        self.assertEqual(len(self.project.files), 2)
        self.assertEqual(self.project.get_file("test_main.py").file_type, "test")

    def test_apply_patch(self):
        patch = DebuggingPatch(
            root_cause_analysis="Fixed greeting string in main.py and added unit tests",
            files_to_update=[
                FileArtifact(
                    filename="main.py",
                    content="print('Hello World')",
                    file_type="source",
                ),
                FileArtifact(
                    filename="test_main.py",
                    content="def test_greeting(): pass",
                    file_type="test",
                ),
            ],
        )
        self.project.apply_patch(patch)

        self.assertEqual(len(self.project.files), 2)
        self.assertEqual(self.project.get_file("main.py").content, "print('Hello World')")


if __name__ == "__main__":
    unittest.main()
