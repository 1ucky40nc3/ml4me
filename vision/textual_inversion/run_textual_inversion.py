from typing import (
    Optional
)

import os
import sys
import math
import random
import logging
import itertools
from dataclasses import (
    dataclass,
    field
)

import numpy as np

import PIL

import torch
import torch.nn.functional as F

import torchvision

import transformers
import diffusers
import accelerate

from tqdm import tqdm


logger = accelerate.logging.get_logger(__name__)


LEARNABLE_CONCEPTS = ['object', 'style']

INTERPOLATION_MAP = {
    'linear': PIL.Image.LINEAR,
    'bilinear': PIL.Image.BILINEAR,
    'bicubic': PIL.Image.BICUBIC,
    'lanczos': PIL.Image.LANCZOS
}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='stabilityai/stable-diffusion-2-1',
        metadata={
            'help': 'The name or path of a pretrained model.'
        }
    )
    model_revision: str = field(
        default='main',
        metadata={
            'help': 'The a specific model version/branch/tag/commit id (e.g. "main").'
        }
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            'help': (
                'Whether to use your huggingface.co authentication token.'
                ' Note: Set it before executing this script via: `>>> huggingface-cli login`.'
            )
        }
    )


@dataclass
class DataArguments:
    data_dir: str = field(
        metadata={
            'help': 'The directory containing training images.'
        }
    )
    initializer_token: str = field(
        metadata={
            'help': 'A keyword that summarizes your new concept.'
        }
    )
    placeholder_token: str = field(
        default='<concept>',
        metadata={
            'help': (
                'A property token to represent the new concept.'
                ' Note: It is advised to use a keyword and special chars as part of this token.'
                ' This is meant to avoid a collision with other tokens in the vocabulary.'
            )
        }
    )
    learnable_concept: str = field(
        default='object',
        metadata={
            'help': f'A property to learn from the data. This can be one of {LEARNABLE_PROPERTIES}."'
        }
    )
    size: int = field(
        default=512,
        metadata={
            'help': 'The with and height dimension of images during training.'
        }
    )
    interpolation: str = field(
        default='bicubic',
        metadata={
            'help': (
                'The interpolation setting during image resizing.'
                f' This can be one of {list(INTERPOLATION_MAP.keys())}'
            )
        }
    )
    flip_p: float = field(
        default=0.5,
        metadata={
            'help': 'The probabiltiy of flipping an image during training.'
        }
    )
    center_crop: bool = field(
        default=False,
        metadata={
            'help': 'Whether to crop images in the center during training.'
        }
    )
    num_repeats: int = field(
        default=100,
        metadata={
            'help': 'The number of times we duplicate the dataset during training.'
        }
    )

    def __post_init__(self) -> None:
        '''Check if the arguments meet expectations.
        
        Raise:
            ValueError: The arguments don't meet expectations.
        '''
        if self.learnable_concept not in LEARNABLE_CONCEPTS:
            raise ValueError(
                f'The `learnable_concept` has to be one of {LEARNABLE_CONCEPTS}!'
                f' "{self.learnable_concept}" not in {LEARNABLE_CONCEPTS}'
            )
        if self.interpolation not in INTERPOLATION_MAP.keys():
            raise ValueError(
                f'The `interpolation` has to be one of {list(INTERPOLATION_MAP.keys())}!'
                f' "{self.interpolation}" not in {list(INTERPOLATION_MAP.keys())}'
            )


@dataclass
class TestingArguments:
    prompt_or_path_to_prompts: Optional[str] = field(
        default=None,
        metadata={
            'help': 'A text prompt or a text file where each line is a new prompt.'
        }
    )


def get_mixed_precision(training_args: transformers.TrainingArgs) -> str:
    '''Return a which precision shall be used.

    Args:
        training_args: Transformers training arguments.
    
    Returns:
        A string representing a mixed precision option.
        This is one of ("no", "fp16", "bf16").
    '''
    mixed_precision = 'no'
    if training_args.fp16:
        mixed_precision = 'fp16'
    elif training_args.bf16:
        mixed_precision = 'bf16'
    return mixed_precision


def main() -> None:
    parser = transformers.HfArgumentParser((
        ModelArguments, 
        DataArguments, 
        TestingArguments,
        transformers.TrainingArguments
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, test_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, test_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = accelerate.Accelerator(
        mixed_precision=get_mixed_precision(training_args),
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=training_args.report_to,
        logging_dir=training_args.logging_dir
    )


if __name__ == '__main__':
    main()