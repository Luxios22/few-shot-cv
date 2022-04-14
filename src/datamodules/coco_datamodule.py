from typing import Optional, Tuple
import os
import random
from tensorflow.keras.utils import get_file
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
from torchvision.datasets import CocoCaptions
from transformers import AutoTokenizer
from src.models.clip_module import Tokenizer, TEXT_MODEL


class COCODataModule(LightningDataModule):
    """Example of LightningDataModule for COCO dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/COCO",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose([transforms.Resize((128, 128)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                        std=(0.229, 0.224, 0.225))
                                ])
        # inv_tfm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
        #                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        #                                 transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
        #                                                     std = [ 1., 1., 1. ]),
        #                             ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.data_dir = data_dir
        self.annotations_dir = os.path.join(data_dir, "annotations")
        self.images_dir = os.path.join(data_dir, "train2014")
        self.annotation_file = os.path.join(data_dir, "captions_train2014.json")

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # Download caption annotation files
        if not os.path.exists(self.annotations_dir):
            annotation_zip = get_file(
                "captions.zip",
                cache_dir=os.path.abspath("."),
                cache_subdir=self.data_dir,
                origin="http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                extract=True,
            )
            os.remove(annotation_zip)

        # Download image files
        if not os.path.exists(self.images_dir):
            image_zip = get_file(
                "train2014.zip",
                cache_dir=os.path.abspath("."),
                cache_subdir=self.data_dir,
                origin="http://images.cocodataset.org/zips/train2014.zip",
                extract=True,
            )
            os.remove(image_zip)

        print("Dataset is downloaded and extracted successfully.")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        tokenizer = Tokenizer(AutoTokenizer.from_pretrained(TEXT_MODEL))
        # load datasets only if they're not loaded already
        target_tfm = lambda x: tokenizer(random.choice(x))

        dataset = CocoCaptions(root = self.images_dir,
                                annFile = self.annotation_file,
                                transform=self.transforms,
                                target_transform=target_tfm,
        )
        # train_len = int(0.8*len(cap))
        # self.data_train, self.data_val = random_split(cap, [train_len, len(cap) - train_len])
        train_val_test_split = [x**len(dataset) for x in self.hparams.train_val_test_split]
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=dataset,
            lengths=train_val_test_split,
            # generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )
