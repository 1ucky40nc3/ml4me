from typing import (
    Any,
    Tuple,
    Optional
)

import os
import sys
import glob
import math
import shutil
import random
import logging
import functools
from dataclasses import (
    dataclass,
    field
)

import numpy as np

import pandas as pd

import torch
import torch.nn.functional as F

import torchvision

import transformers
import datasets
import diffusers
import accelerate

from tqdm import tqdm


logger = accelerate.logging.get_logger(__name__)


LEARNABLE_CONCEPTS = ['object', 'style']


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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'A cache dir for models (and datasets) from "huggingface.co".'
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
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The name of a dataset on "huggingface.co".'
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            'help': 'A config name for the given dataset from "huggingface.co".'
        }
    )
    image_column: Optional[str] = field(
        default='image_path',
        metadata={
            'help': 'The name of the dataset column containing the image paths.'
        },
    )
    template_column: Optional[str] = field(
        default='template',
        metadata={
            'help': 'The name of the dataset column containing the templates.'
        }
    )
    train_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'A directory containing the training dataset.'
        }
    )
    images_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The directory containing training images.'
        }
    )
    templates_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': (
                'The directory containing template files.'
                ' Note: We concatenate the contents of theese files.'
                ' Each line inside theese files is a new template'
                ' and has to contain the "{}" substring as placeholder for `placeholder_token`.'
            )
        }
    )
    initializer_token: Optional[str] = field(
        default=None,
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
            'help': f'A property to learn from the data. This can be one of {LEARNABLE_CONCEPTS}."'
        }
    )
    size: int = field(
        default=512,
        metadata={
            'help': 'The with and height dimension of images during training.'
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
    data_output_dir: str = field(
        default='./data',
        metadata={
            'help': 'A path where we output the training dataset.'
        }
    )
    overwrite_data_output_dir: bool = field(
        default=False,
        metadata={
            'help': 'Whether to overwrite the training dataset output directory.'
        }
    )

    def __post_init__(self) -> None:
        '''Check if the arguments meet expectations.
        
        Raise:
            ValueError: The arguments don't meet expectations.
        '''
        if self.initializer_token is None:
            raise ValueError(
                f"The `initializer_token` can't be `None`!"
                ' You have to set a `--initializert_token` argument.'
            )
        if self.learnable_concept not in LEARNABLE_CONCEPTS:
            raise ValueError(
                f'The `learnable_concept` has to be one of {LEARNABLE_CONCEPTS}!'
                f' "{self.learnable_concept}" not in {LEARNABLE_CONCEPTS}'
            )
        if (os.path.isdir(self.data_output_dir) and len(os.listdir(self.data_output_dir)) > 0) and not self.overwrite_data_output_dir:
            raise ValueError(
                f'The `data_output_dir` exists and is not empty!'
                ' Set a different `data_output_dir` of use `--overwrite_data_output_dir`!'
                f' Directory: {self.data_output_dir}'
            )
            

@dataclass
class TestingArguments:
    prompt_or_path_to_prompts: Optional[str] = field(
        default=None,
        metadata={
            'help': 'A text prompt or a text file where each line is a new prompt.'
        }
    )


def get_mixed_precision(training_args: transformers.TrainingArguments) -> str:
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


def load_tokenizer(
    model_args: ModelArguments, 
    data_args: DataArguments,
    initializer_token: str,
    placeholder_token: str = '<placeholder>',
) -> Tuple[transformers.CLIPTokenizer, int, int]:
    '''Return the prepared pretrained tokenizer and relevant token ids.
    
    Args:
        model_args: A `ModelArguments` object.
        data_args: A `DataArguments` object.
        initializer_token: A string/keyword as summary of your concept.
        placeholder_token: A placeholder token for the new concept.
    
    Returns:
        A tuple of the CLIP tokenizer (`transformers.CLIPTokenizer`)
        and the token ids (`int`) of the initlizer and placeholder tokens.
    
    Raises:
        ValueError: The `placeholder_token` or `initializer_token` args are invalid.
    '''
    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        model_args.model_name_or_path,
        subfolder="tokenizer",
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        revision=model_args.model_revision
    )

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            'The tokenizer already contains the placeholder token.'
            f' Provide an alternative! Placeholder token: "{placeholder_token}"'
        )
    
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError(
            'The tokenizer has the yield one token id for `initializer_token` but did > 1!'
            ' Provide a different `initializer_token`. Note: Short keywords might be good.'
        )
    
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    return tokenizer, initializer_token_id, placeholder_token_id


def deep_getattr(obj: Any, path: str) -> Any:
    '''Get the last object attribute along the path.

    Args:
        obj: Any python object with attributes along the path.
        path: A string with attributes seperated by the "." char.
    
    Returns:
        The last attribute along the path.

    Raises:
        AttributeError: An attribute along the path doesn't exist.
    '''
    attr = obj
    for name in path.split('.'):
        attr = getattr(attr, name)
    return attr


def freeze_model(model: torch.nn.Module, modules: Optional[Tuple[str]] = None) -> None:
    '''Freeze a model.

    Args:
        model: A torch model.
        modules: A list of modules in the model. 
                 Only theese modules will be frozen.
    
    Raises:
        AttributeError: A module of `modules` doesn't exist.
    '''
    if modules is not None:
        for module in modules:
            modules = deep_getattr(module)
            for param in module.parameters():
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False


def load_models(
    model_args: ModelArguments, 
    tokenizer: transformers.CLIPTokenizer,
    initializer_token_id: int,
    placeholder_token_id: int
) -> Tuple[transformers.CLIPTextModel, diffusers.AutoencoderKL, diffusers.UNet2DConditionModel]:
    '''Return a prepared pretrained Stable-Diffusion model.

    Args:
        model_args: A `ModelArguments` object.
        tokenizer: A prepared and pretrained `transformers.CLIPTokenizer`.
        initializer_token_id: The id of the tokenized `initializer_token`.
        placeholder_token_id: The id of the tokenized `placeholder_token`.

    Returns:
        A tuple of a pretrained CLIP text model (`transformers.CLIPTextModel`), 
        a pretrained VAE (`diffusers.AutoencoderKL`) and a pretrained UNET (`diffusers.UNet2DConditionModel`).
        All of theese models are frozen at specific places and thus prepared for training. 

    Raises:
        AttributeError: The CLIP text model has unexpected components.
    '''
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        model_args.model_name_or_path,
        subfolder='text_encoder',
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        revision=model_args.model_revision
    )
    vae = diffusers.AutoencoderKL.from_pretrained(
        model_args.model_name_or_path,
        subfolder='vae',
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        revision=model_args.model_revision
    )
    unet = diffusers.UNet2DConditionModel.from_pretrained(
        model_args.model_name_or_path,
        subfolder='unet',
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        revision=model_args.model_revision
    )

    text_encoder.resize_token_embeddings(len(tokenizer))

    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    freeze_model(
        text_encoder,
        modules=[
            'text_model.encoder',
            'text_model.final_layer_norm',
            'text_model.embeddings.position_embedding'
        ]
    )
    freeze_model(vae)
    freeze_model(unet)

    return text_encoder, vae, unet


def clean_template(string: str) -> str:
    '''Clean a template string.

    Get rid of unwanted chars or sequences like:
    - "\t"
    - "\n"
    - ...

    Args:
        string: The template string.

    Returns:
        The cleaned template string.
    '''
    string = string.replace('\t', '').replace('\r', '').replace('\n', '')
    return string


def get_extension(path: str) -> str:
    '''Return a file's extension.

    Args:
        path: A file path.
    
    Returns:
        The file's extension.
    '''
    _, ext = os.path.splitext(path)
    return ext


def create_data(data_args: DataArguments) -> None:
    '''Create a ImageFolder dataset.

    Args:
        data_args: A `DataArguments` object.

    Returns:
        The path to the dataset directory. 
    '''
    if data_args.overwrite_data_output_dir and os.path.exists(data_args.data_output_dir):
        shutil.rmtree(data_args.data_output_dir)
    os.makedirs(data_args.data_output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(data_args.images_dir, '*'))
    template_files = glob.glob(os.path.join(data_args.templates_dir, '*.txt'))

    templates = []
    for template_file in template_files:
        with open(template_file, 'r', encoding='utf-8') as f:
            templates.extend(f.readlines())
        
    data = []
    num_samples = len(image_files) * data_args.num_repeats
    for i in range(num_samples):
        template = random.choice(templates)
        template = template.format(data_args.placeholder_token)
        template = clean_template(template)
        image_idx = i - (i // len(image_files)) * len(image_files)
        image = image_files[image_idx]
        copy_file = f'%0{len(str(num_samples))}d{get_extension(image)}' % i
        copy_path = os.path.join(data_args.data_output_dir, copy_file)
        shutil.copy(image, copy_path)
        data.append([copy_file, template])
    
    metadata_file = os.path.join(data_args.data_output_dir, 'metadata.tsv')
    df = pd.DataFrame(data)
    df.to_csv(
        metadata_file,
        sep='\t',
        header=[data_args.image_column, data_args.template_column],
        index=False,
        encoding='utf-8'
    )


def load_data(
    data_args: DataArguments, 
    model_args: ModelArguments, 
    training_args: transformers.TrainingArguments,
    accelerator: accelerate.Accelerator,
    tokenizer: transformers.CLIPTokenizer
) -> Tuple[Optional[torch.utils.data.DataLoader]]:
    '''Return a training data loader.

    Args:
        data_args: A `DataArguments` object.
        model_args: A `ModelArguments` object.
        training_args: A `transformers.TrainingArguments` object.
        accelerator: A `accelerate.Accelerator` object.
        tokenizer: A `transformers.CLIPTokenizer` object.

    Returns:
        A data loader for training purposes.

    Raises:
        ValueError: Requirements to load or create a dataset weren't met.
    '''
    if data_args.dataset_name is not None:
        dataset = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None
        )
    elif data_args.train_dir is not None:
        data_files = {'train': os.path.join(data_args.train_dir, '**')}
        dataset = datasets.load_dataset(
            'imagefolder',
            data_files=data_files,
            cache_dir=model_args.cache_dir
        )
    elif (
        data_args.images_dir is not None
        and data_args.templates_dir is not None
    ):
        create_data(data_args)
        data_files = {'train': os.path.join(data_args.data_output_dir, '**')}
        dataset = datasets.load_dataset(
            'imagefolder',
            data_files=data_files,
            cache_dir=model_args.cache_dir
        )
    else:
        raise ValueError(
            'You either need to specify a `dataset_name`, a `train_dir` or create a new dataset.'
            ' To create a new dataset you need to set `images_dir` and `templates_dir`.'
        )
    
    column_names = dataset['train'].column_names
    
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(data_args.size),
        torchvision.transforms.RandomHorizontalFlip(data_args.flip_p),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda img: img * 2 - 1)
    ])

    def transform_fn(examples, transforms):
        '''Apply transforms on images of a batch.'''
        examples["pixel_values"] = [
            transforms(pil_img.convert("RGB"))
            for pil_img in examples['image']
        ]
        return examples
    
    train_transform_fn = functools.partial(transform_fn, transforms=train_transforms)
    
    def tokenize_fn(examples):
        '''Tokenize the templates of a batch'''
        templates = list(examples[data_args.template_column])
        inputs = tokenizer(
            templates, 
            padding='max_length', 
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors='pt'
        )
        examples['input_ids'] = inputs.input_ids
        return examples

    if training_args.do_train:
        with accelerator.main_process_fist():
            dataset['train'] = dataset['train'].shuffle(seed=training_args.seed)
            train_dataset = train_dataset.map(
                function=tokenize_fn,
                batched=True,
                remove_columns=[col for col in column_names if col != 'image'],
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc='Running tokenizer on train dataset',
            )
            train_dataset.set_transform(train_transform_fn)

    def collate_fn(examples):
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        input_ids = torch.tensor([example['input_ids'] for example in examples])
        return {'pixel_values': pixel_values, 'input_ids': input_ids}

    if training_args.do_train:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=training_args.batch_size
        )

    return (
        train_dataloader if training_args.do_train else None,
    )

    


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
        model_args, data_args, testing_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, testing_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = accelerate.Accelerator(
        mixed_precision=get_mixed_precision(training_args),
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=training_args.report_to,
        logging_dir=training_args.logging_dir
    )


if __name__ == '__main__':
    main()