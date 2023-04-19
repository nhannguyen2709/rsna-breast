from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    image_folder: str = field(default="data/crop_images", metadata={"help": "Image folder"})
    use_external: bool = field(default=False, metadata={"help": "Use external data"})
    external_csvs: str = field(default="data/train_vindr.csv")
    external_image_folders: str = field(
        default="data/vindr_images", metadata={"help": "External image folder"}
    )
    height: int = field(default=512, metadata={"help": "Image height"})
    width: int = field(default=512, metadata={"help": "Image width"})
    fold: int = field(default=0, metadata={"help": "Fold"})
    pn_ratio: int = field(default=1, metadata={"help": "Positive/negative ratio"})
