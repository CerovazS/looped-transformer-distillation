from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

from loopdistill.teachers.base import TeacherOutput, TeacherRunner


class ExternalRepoTeacher(TeacherRunner):
    """Thin adapter for local Attractor/Parcae-style repositories.

    The external class must expose `run_batch(tokens, attention_mask, depths)` returning either
    `TeacherOutput` or a dictionary with the same fields.
    """

    def __init__(
        self,
        teacher_id: str,
        repo_path: str | None,
        module_path: str | None,
        class_name: str | None,
        checkpoint_path: str | None = None,
        tokenizer_id: str | None = None,
        **kwargs,
    ):
        self.teacher_id = teacher_id
        self.repo_path = repo_path
        self.module_path = module_path
        self.class_name = class_name
        self.checkpoint_path = checkpoint_path
        self.tokenizer_id = tokenizer_id
        self.kwargs = kwargs
        self._runner = None

    def _load_runner(self):
        if self._runner is not None:
            return self._runner
        if not self.repo_path or not self.module_path or not self.class_name:
            raise RuntimeError(
                f"{self.teacher_id} adapter requires repo_path, module_path, and class_name."
            )
        repo = Path(self.repo_path).expanduser().resolve()
        sys.path.insert(0, str(repo))
        module = importlib.import_module(self.module_path)
        cls = getattr(module, self.class_name)
        self._runner = cls(checkpoint_path=self.checkpoint_path, tokenizer_id=self.tokenizer_id, **self.kwargs)
        return self._runner

    def run_batch(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        depths: list[int],
    ) -> TeacherOutput:
        runner = self._load_runner()
        output = runner.run_batch(tokens=tokens, attention_mask=attention_mask, depths=depths)
        if isinstance(output, TeacherOutput):
            return output
        return TeacherOutput(**output)
