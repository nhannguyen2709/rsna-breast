from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments relating to model.
    """

    model_name: str = field(default="base", metadata={"help": "Model name"})
    resume: Optional[str] = field(default=None, metadata={"help": "Path of model checkpoint"})
    swa: Optional[bool] = field(default=False, metadata={"help": "Average checkpoints"})
    resume_swa: Optional[str] = field(
        default=None, metadata={"help": "Path(s) of model checkpoint(s)"}
    )
    resume_single: str = field(
        default="", metadata={"help": "Path of single image model checkpoint"}
    )
