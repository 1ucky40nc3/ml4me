from typing import (
    Any,
    List,
    Dict,
    Tuple,
    Union,
    Optional,
    Callable
)

import os
import sys
import glob
import math
import json
import uuid
import copy
import shutil
import random
import logging
import functools
from dataclasses import (
    dataclass,
    field,
    asdict
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

from tqdm.auto import tqdm

import PIL


logger = accelerate.logging.get_logger(__name__)


WandbConfig = Any


TRAINING_ARGS_NAME = 'training_args.bin'
TRAINER_STATE_NAME = 'trainer_state.json'
KNOCKKNOCK_DISCORD_WEBHOOK_URL_ENV_VARIABLE = 'KNOCKKNOCK_DISCORD_WEBHOOK_URL'


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
    train_file: Optional[str] = field(
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
    rand_augment_num_ops: Optional[int] = field(
        default=2,
        metadata={
            'help': 'Number of random augmentation transforms during training.'
        }
    )
    rand_augment_magnitude: Optional[int] = field(
        default=9,
        metadata={
            'help': 'The magnitude of random augmentation transforms during training.'
        }
    )
    rand_augment_num_magnitude_bins: Optional[int] = field(
        default=31,
        metadata={
            'help': 'The number magnitude values of random augmentation transforms during training.'
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
    overwrite_cache: bool = field(
        default=False, 
        metadata={
            'help': 'Whether to overwrite cached/preprocessed datasets.'
        }
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            'help': 'The number of workers during preprocessing.'
        }
    )


@dataclass
class OurTrainingArguments:
    do_not_force_save: bool = field(
        default=False,
        metadata={
            'help': 'Whether to not force a save on the end of a training run.'
        }
    )
    do_not_save_pipeline: bool = field(
        default=False,
        metadata={
            'help': 'Whether to not save the diffusion pipeline after a training run.'
        }
    )
            

@dataclass
class InferenceArguments:
    do_inference: bool = field(
        default=False,
        metadata={
            'help': 'Whether to do inference.'
        }
    )
    inference_output_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': (
                'An optional output directory for inference results.'
                ' We append this directory to the training arguments `--output_directory`.'
            )
        }
    )
    prompt_or_path_to_prompts: Optional[str] = field(
        default=None,
        metadata={
            'help': 'A text prompt or a text file where each line is a new prompt.'
        }
    )
    height: Optional[int] = field(
        default=None,
        metadata={
            'help': (
                'The height of generated images.'
                ' This defaults to `self.unet.config.sample_size * self.vae_scale_factor`.'
            )
        }
    )
    width: Optional[int] = field(
        default=None,
        metadata={
            'help': (
                'The width of generated images.'
                ' This defaults to `self.unet.config.sample_size * self.vae_scale_factor`.'
            )
        }
    )
    num_inference_steps: int = field(
        default=30,
        metadata={
            'help': 'The number inference (denoising) steps during image generation.'
        }
    )
    guidance_scale: float = field(
        default=7.5,
        metadata={
            'help': 'The guidance scale for classifier-free diffusion guidance.'
        }
    )
    num_images_per_prompt: int = field(
        default=1,
        metadata={
            'help': 'The number of samples to generate.'
        }
    )
    apply_token_merging: bool = field(
        default=False,
        metadata={
            'help': 'Whether to apply token merging (ToMe) (see https://github.com/dbolya/tomesd).'
        }
    )
    tome_ratio: float = field(
        default=0.5,
        metadata={
            'help': (
                'The ratio of tokens to merge during ToMe.'
                ' This only applies if `apply_token_merging == True`.'
            )
        }
    )
    tome_max_downsample: int = field(
        default=1,
        metadata={
            'help': (
                'Only apply ToMe with layers'
                ' that do less or equal than this amount of downsampling.'
                ' This value should be one of [1, 2, 4, or 8].'
            )
        }
    )


@dataclass
class SweepArguments:
    do_sweep: bool = field(
        default=False,
        metadata={
            'help': 'Whether to do a hyperparameter sweep with wandb.'
        }
    )
    sweep_config_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The path to a JSON config file for wandb hyperparameter sweeps.'
        }
    )
    sweep_count: Optional[int] = field(
        default=None,
        metadata={
            'help': 'The number of runs per sweep.'
        }
    )


@dataclass
class KnockKnockArguments:
    knockknock_on_discord: bool = field(
        default=False,
        metadata={
            'help': 'Whether send `knockknock` notifications on discord.'
        }
    )
    knockknock_discord_webhook_url: Optional[str] = field(
        default=None,
        metadata={
            'help': (
                'A discord webhook url. See "https://github.com/huggingface/knockknock" for instructions.'
                ' The webhook url value can be specified w/ this arg or the `KNOCKKNOCK_DISCORD_WEBHOOK_URL` env variable.'
            )
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


def get_weight_dtype(mixed_precision: Optional[str]) -> torch.dtype:
    '''Get the `torch.dtype` associated with the precision.

    Args:
        mixed_precision: A mixed precision string from `get_mixed_precision(...)`.
    
    Returns:
        The `torch.dtype` associated with the precision
    '''
    weight_dtype = torch.float32
    if mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    return weight_dtype


def load_tokenizer(
    model_args: ModelArguments, 
    data_args: DataArguments,
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
    if data_args.initializer_token is None:
        raise ValueError(
            f"The `initializer_token` can't be `None`!"
            ' You have to set a `--initializert_token` argument.'
        )

    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        model_args.model_name_or_path,
        subfolder='tokenizer',
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        revision=model_args.model_revision
    )

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(data_args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            'The tokenizer already contains the placeholder token.'
            f' Provide an alternative! Placeholder token: "{data_args.placeholder_token}"'
        )
    
    token_ids = tokenizer.encode(data_args.initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError(
            'The tokenizer has the yield one token id for `initializer_token` but did > 1!'
            ' Provide a different `initializer_token`. Note: Short keywords might be good.'
        )
    
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(data_args.placeholder_token)

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
            module = deep_getattr(model, module)
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


def load_noise_scheduler(model_args: ModelArguments) -> diffusers.DDPMScheduler:
    '''Load a noise scheduler for the denoising process.

    Args:
        model_args: A `ModelArguments` object.

    Returns:
        A 'diffusers.DDPMScheduler' initialized with the pretrained model config.
    '''
    return diffusers.DDPMScheduler.from_config(
        model_args.model_name_or_path, 
        subfolder='scheduler'
    ) 


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
    
    Raises:
        ValueError: Overwrite or create new `data_args.data_output_dir`.
    '''
    if (
        os.path.isdir(data_args.data_output_dir) 
        and len(os.listdir(data_args.data_output_dir)) > 0
    ) and not data_args.overwrite_data_output_dir:
        raise ValueError(
            f'The `data_output_dir` exists and is not empty!'
            ' Set a different `data_output_dir` of use `--overwrite_data_output_dir`!'
            f' Directory: {data_args.data_output_dir}'
        )

    if data_args.overwrite_data_output_dir and os.path.exists(data_args.data_output_dir):
        shutil.rmtree(data_args.data_output_dir)
    os.makedirs(data_args.data_output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(data_args.images_dir, '*'))
    template_files = glob.glob(os.path.join(data_args.templates_dir, '*.txt'))

    templates = []
    for template_file in template_files:
        with open(template_file, 'r', encoding='utf-8') as f:
            templates.extend(f.readlines())

    image_copies = []
    num_decimals = len(str(len(image_files)))
    for i, image_file in enumerate(image_files):
        copy_file = f'%0{num_decimals}d{get_extension(image_file)}' % i
        copy_path = os.path.join(data_args.data_output_dir, copy_file)
        shutil.copy(image_file, copy_path)
        image_copies.append(copy_path)

    data = []
    num_samples = len(image_files) * data_args.num_repeats
    for i in range(num_samples):
        template = random.choice(templates)
        template = template.format(data_args.placeholder_token)
        template = clean_template(template)
        image_idx = i - (i // len(image_files)) * len(image_files)
        image = image_copies[image_idx]
        data.append([image, template])
    
    metadata_file = os.path.join(data_args.data_output_dir, 'metadata.csv')
    df = pd.DataFrame(data)
    df.to_csv(
        metadata_file,
        header=[data_args.image_column, data_args.template_column],
        index=False,
        encoding='utf-8'
    )


def read_image(path: str) -> torch.Tensor:
    '''Read a image file.

    Args:
        path: The image file path.
    
    Returns:
        A `torch.Tensor` with the RGB image pixel data.
        The pixel values are from 0 to 255 in the `torch.uint8` format.
    '''
    return torchvision.io.read_image(
        path, 
        mode=torchvision.io.ImageReadMode.RGB
    )


def load_data(
    data_args: DataArguments, 
    model_args: ModelArguments, 
    training_args: transformers.TrainingArguments,
    accelerator: accelerate.Accelerator,
    tokenizer: transformers.CLIPTokenizer,
    batch_size: int
) -> Tuple[Optional[torch.utils.data.DataLoader]]:
    '''Return a training data loader.

    Args:
        data_args: A `DataArguments` object.
        model_args: A `ModelArguments` object.
        training_args: A `transformers.TrainingArguments` object.
        accelerator: A `accelerate.Accelerator` object.
        tokenizer: A `transformers.CLIPTokenizer` object.
        batch_size: The number of samples on each device per mini-batch.

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
    elif data_args.train_file is not None:
        data_files = {'train': data_args.train_file}
        dataset = datasets.load_dataset(
            get_extension(data_args.train_file)[1:],
            data_files=data_files,
            cache_dir=model_args.cache_dir
        )
    elif (
        data_args.images_dir is not None
        and data_args.templates_dir is not None
    ):
        create_data(data_args)
        data_files = {'train': os.path.join(data_args.data_output_dir, 'metadata.csv')}
        dataset = datasets.load_dataset(
            get_extension(data_files['train'])[1:],
            data_files=data_files,
            cache_dir=model_args.cache_dir
        )
    else:
        raise ValueError(
            'You either need to specify a `dataset_name`, a `train_dir` or create a new dataset.'
            ' To create a new dataset you need to set `images_dir` and `templates_dir`.'
        )
    
    column_names = dataset['train'].column_names

    class TrainTransform(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transforms = torch.nn.Sequential(
                torchvision.transforms.RandomResizedCrop(data_args.size, antialias=True),
                torchvision.transforms.RandomHorizontalFlip(data_args.flip_p),
                torchvision.transforms.RandAugment(
                    data_args.rand_augment_num_ops, 
                    data_args.rand_augment_magnitude, 
                    data_args.rand_augment_num_magnitude_bins
                ),
                torchvision.transforms.ConvertImageDtype(torch.float),
            )

        def forward(self, x) -> torch.Tensor:
            with torch.no_grad():
                x = self.transforms(x)
                x = x / 127.5 - 1.0
            return x

    def transform_fn(examples, transform):
        '''Apply transforms on images of a batch.'''
        examples['pixel_values'] = [
            transform(read_image(image_path))
            for image_path in examples[data_args.image_column]
        ]
        return examples
    
    train_transform = torch.jit.script(TrainTransform())
    train_transform_fn = functools.partial(transform_fn, transform=train_transform)
    
    def tokenize_fn(examples):
        '''Tokenize the templates of a batch'''
        templates = list(examples[data_args.template_column])
        inputs = tokenizer(
            templates, 
            padding='max_length', 
            max_length=tokenizer.model_max_length,
            truncation=True
        )
        examples['input_ids'] = inputs.input_ids
        return examples

    if training_args.do_train:
        with accelerator.main_process_first():
            train_dataset = dataset['train'].shuffle(seed=training_args.seed)
            train_dataset = train_dataset.map(
                function=tokenize_fn,
                batched=True,
                remove_columns=[col for col in column_names if col != data_args.image_column],
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc='Running tokenizer on train dataset',
            )
            # Transform images on the fly
            train_dataset.set_transform(train_transform_fn)

    def collate_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        input_ids = torch.tensor([example['input_ids'] for example in examples], dtype=torch.long)
        return {'pixel_values': pixel_values, 'input_ids': input_ids}

    if training_args.do_train:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=batch_size
        )

    return (
        train_dataloader if training_args.do_train else None,
    )


def get_global_step(path: str) -> int:
    '''Get the global step from the checkpoint name.

    Args:
        path: The path of the checkpoint directory.
              The directory name follows the f"step_{step}" format.

    Returns:
        The global step as saved in the directory name.
    '''
    basename = os.path.basename(path)
    filename, _ = os.path.splitext(basename)
    global_step = int(filename.replace('step_', ''))
    return global_step


def get_num_trainable_parameters(models: Union[torch.nn.Module, List[torch.nn.Module]] = []) -> int:
    '''Return the number of trainable parameters.

    Args:
        models: A model or list of models of type `toch.nn.Module`.

    Returns:
        The total number of trainable parameters.
    '''
    if not isinstance(models, (list, tuple)):
        models = [models]

    num_trainable_parameters = 0
    for model in models:
        num_trainable_parameters += sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_trainable_parameters


def get_save_steps(
    training_args: transformers.TrainingArguments,
    num_update_steps_per_epoch: Optional[int] = None
) -> int:
    '''Number of update steps per save interval.

    Args:
        training_args: A `transformers.TrainingArguments` object.
        num_update_steps_per_epoch: The number of update steps per epoch.
                                    This kwarg is required if 
                                    `training_args.save_stragety == "epoch"`.
    
    Returns:
        The number of update steps per save interval.

    Raises:
        ValueError: The `num_update_steps_per_epoch` is required when `training_args.save_stragety == "epoch"`.
    '''
    if training_args.save_strategy == 'steps':
        return training_args.save_steps
    else:
        if num_update_steps_per_epoch is None:
            raise ValueError('The `num_update_steps_per_epoch` is required when `training_args.save_stragety == "epoch"`!')
        return training_args.save_steps * num_update_steps_per_epoch



def maybe_save(
    training_args: transformers.TrainingArguments, 
    accelerator: accelerate.Accelerator,
    global_step: int,
    num_update_steps_per_epoch: Optional[int] = None,
    force_save: bool = False
) -> None:
    '''Maybe save if conditions are met.

    Check if the `training_args.save_strategy` and `training_args.save_steps`
    specify conditions that are met at the current `global_step`.

    Args:
        training_args: A `transformers.TrainingArguments` object.
        accelerator: Our current accelerator object.
        global_step: The current global training step.
        num_update_steps_per_epoch: The number of update steps per epoch.
                                    This kwarg is required if 
                                    `training_args.save_stragety == "epoch"`.
        force_save: Whether to saving a checkpoint once.
   
    Raises:
        ValueError: The `num_update_steps_per_epoch` is required when `training_args.save_stragety == "epoch"`.
    '''
    save_steps = get_save_steps(training_args, num_update_steps_per_epoch)
    if (global_step % save_steps == 0 and global_step != 0) or force_save:
        if training_args.save_total_limit is not None:
            # Only keep the latest `training_args.save_total_limit` number of checkpoints
            all_saves = glob.glob(os.path.join(training_args.output_dir, r'steps_\d'))
            all_saves.sort()
            if len(all_saves) > training_args.save_total_limit:
                num_delete = len(all_saves) - training_args.save_total_limit
                for save_dir in all_saves[:num_delete]:
                    shutil.rmtree(save_dir)

        save_dir = os.path.join(training_args.output_dir, f'steps_{global_step}')
        accelerator.save_state(save_dir)


def save_pipeline(
    model_args: ModelArguments,
    training_args: transformers.TrainingArguments,
    accelerator: accelerate.Accelerator,
    text_encoder: transformers.CLIPTextModel, 
    vae: diffusers.AutoencoderKL, 
    unet: diffusers.UNet2DConditionModel,
    tokenizer: transformers.CLIPTokenizer
) -> None:
    '''Create and save a `diffusers.StableDiffusionPipeline`.

    Args:
        training_args: A `transformers.TrainingArguments` object.
        accelerator: Our current accelerator object.
        text_encoder: A pretrained `transformers.CLIPTextModel`.
        vae: A pretrained `diffusers.AutoencoderKL`.
        unet: A pretrained `diffusers.UNet2DConditionModel`.
        tokenizer: A `transformers.CLIPTokenizer` object.
    '''
    pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
        model_args.model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        vae=vae,
        unet=unet,
    )
    pipeline.save_pretrained(training_args.output_dir)


def load_pipeline(
    inference_args: InferenceArguments,
    training_args: transformers.TrainingArguments,
    accelerator: accelerate.Accelerator,
) -> diffusers.StableDiffusionPipeline:
    '''Create and save a `diffusers.StableDiffusionPipeline`.

    Args:
        inference_args: A `InferenceArguments` object.
        training_args: A `transformers.TrainingArguments` object.
        accelerator: Our current accelerator object.

    Returns:
        A pretrained `diffusers.StableDiffusionPipeline`.
    '''
    noise_scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
        training_args.output_dir, 
        subfolder='scheduler'
    )
    weight_dtype = get_weight_dtype(accelerator.mixed_precision)
    pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
        training_args.output_dir,
        scheduler=noise_scheduler,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    if inference_args.do_inference:
        try:
            import tomesd
            
            tomesd.apply_patch(
                pipeline, 
                ratio=inference_args.tome_ratio,
                max_downsample=inference_args.tome_max_downsample
            )
        except ImportError as e:
            raise ImportError(
                f'{e.__class__.__name__}: If want to apply token merging (ToMe)'
                ' follow the installation instructions from "https://github.com/dbolya/tomesd".'
            )

    return pipeline


def get_logging_steps(
    training_args: transformers.TrainingArguments,
    num_update_steps_per_epoch: Optional[int] = None
) -> int:
    '''Number of update steps per save interval.

    Args:
        training_args: A `transformers.TrainingArguments` object.
        num_update_steps_per_epoch: The number of update steps per epoch.
                                    This kwarg is required if 
                                    `training_args.logging_strategy == "epoch"`.
    
    Returns:
        The number of update steps per save interval.

    Raises:
        ValueError: The `num_update_steps_per_epoch` is required when `training_args.logging_strategy == "epoch"`.
    '''
    if training_args.logging_strategy == 'steps':
        return training_args.logging_steps
    else:
        if num_update_steps_per_epoch is None:
            raise ValueError('The `num_update_steps_per_epoch` is required when `training_args.logging_strategy == "epoch"`!')
        return training_args.logging_steps * num_update_steps_per_epoch


def convert_to_primitives(**kwargs) -> Dict[str, Any]:
    '''Convert `kwargs` into a primitives friendly format.

    This function is used to convert metrics with `torch.Tensor`
    values into a logging friendly format based on primitives.

    Returns:
        A dictionary of the formatted `kwargs`.
    '''
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            kwargs[key] = value.detach().cpu().item()
        elif isinstance(value, (list, tuple, dict)):
            kwargs[key] = json.dumps(value)
    return kwargs


def maybe_log(
    training_args: transformers.TrainingArguments,
    accelerator: accelerate.Accelerator,
    global_step: int,
    metrics: Dict[str, Any] = {},
    progress_bar: Optional[tqdm] = None,
    num_update_steps_per_epoch: Optional[int] = None,
    force_log: bool = False
) -> None:
    '''Maybe log if conditions are met.

    Check if the `training_args.logging_strategy` and `training_args.logging_steps`
    specify conditions that are met at the current `global_step`.

    Args:
        training_args: A `transformers.TrainingArguments` object.
        accelerator: Our current accelerator object.
        global_step: The current global training step.
        num_update_steps_per_epoch: The number of update steps per epoch.
                                    This kwarg is required if 
                                    `training_args.logging_strategy == "epoch"`.
        force_log: Whether to force logging once.

    Raises:
        ValueError: The `num_update_steps_per_epoch` is required when `training_args.logging_strategy == "epoch"`.
    '''
    logging_steps = get_logging_steps(training_args, num_update_steps_per_epoch)
    if global_step % logging_steps == 0 or force_log:
        metrics = convert_to_primitives(**metrics)
        if training_args.report_to is not None:
            accelerator.log(
                {
                    **metrics,
                    'global_step': global_step
                },
                step=global_step
            )
        if progress_bar is not None:
            progress_bar.set_postfix(**metrics)


def maybe_log_or_save(
    training_args: transformers.TrainingArguments,
    accelerator: accelerate.Accelerator,
    global_step: int,
    metrics: Dict[str, Any] = {},
    progress_bar: Optional[tqdm] = None,
    num_update_steps_per_epoch: Optional[int] = None,
    force_log: bool = False,
    force_save: bool = False
) -> None:
    '''Maybe log or save if conditions are met.

    Args:
        training_args: A `transformers.TrainingArguments` object.
        accelerator: Our current accelerator object.
        global_step: The current global training step.
        num_update_steps_per_epoch: The number of update steps per epoch.
                                    This kwarg is required if 
                                    `training_args.save_stragety == "epoch"`.
                                    or `training_args.logging_strategy == "epoch"`.
        force_log: Whether to force logging once.
        force_save: Whether to saving a checkpoint once.
    '''
    maybe_log(
        training_args, 
        accelerator, 
        global_step, 
        metrics, 
        progress_bar, 
        num_update_steps_per_epoch,
        force_log=force_log
    )
    maybe_save(
        training_args, 
        accelerator, 
        global_step, 
        num_update_steps_per_epoch,
        force_save=force_save
    )


def train_fn(
    model_args: ModelArguments,
    data_args: DataArguments,
    our_training_args: OurTrainingArguments,
    training_args: transformers.TrainingArguments
) -> None:
    accelerator = accelerate.Accelerator(
        mixed_precision=get_mixed_precision(training_args),
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=training_args.report_to,
        logging_dir=training_args.logging_dir
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if training_args.report_to is not None:
        config = {
            **asdict(model_args),
            **asdict(data_args),
            **asdict(training_args)
        }
        config = convert_to_primitives(**config)
        accelerator.init_trackers(training_args.run_name, config)
    
    @accelerate.utils.find_executable_batch_size(starting_batch_size=training_args.per_device_train_batch_size)
    def _train_fn(batch_size: int) -> None:
        nonlocal accelerator

        gradient_accumulation_steps = training_args.per_device_train_batch_size // batch_size
        accelerator.gradient_accumulation_steps = gradient_accumulation_steps

        accelerator.free_memory()
        accelerate.utils.set_seed(training_args.seed)

        tokenizer, initializer_token_id, placeholder_token_id = load_tokenizer(
            model_args, data_args
        )
        num_tokens = len(tokenizer)
        text_encoder, vae, unet = load_models(
            model_args, tokenizer, initializer_token_id, placeholder_token_id
        )
        text_encoder = text_encoder.to(accelerator.device)
        weight_dtype = get_weight_dtype(accelerator.mixed_precision)
        vae.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)

        optimizer = torch.optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),
            lr=training_args.learning_rate
        )

        (train_dataloader, *_) = load_data(
            data_args,
            model_args,
            training_args,
            accelerator,
            tokenizer,
            batch_size
        )

        text_encoder, optimizer, train_dataloader = accelerator.prepare(
            text_encoder, optimizer, train_dataloader
        )

        vae.eval()
        unet.train()

        noise_scheduler = load_noise_scheduler(model_args)

        num_steps_per_epoch = len(train_dataloader)
        num_update_steps_per_epoch = max(math.ceil(num_steps_per_epoch / gradient_accumulation_steps), 1)
        if training_args.max_steps > 0:
            max_steps = training_args.max_steps
            num_train_epochs = max_steps // num_update_steps_per_epoch + int(max_steps // num_update_steps_per_epoch > 0)
            num_train_samples = max_steps * training_args.per_device_train_batch_size
        else:
            num_train_epochs = int(training_args.num_train_epochs)
            max_steps = math.ceil(num_train_epochs * num_update_steps_per_epoch)
            num_train_samples = len(train_dataloader) * training_args.per_device_train_batch_size * num_train_epochs

        total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes

        logger.info('***** Running training *****')
        logger.info(f'  Num examples = {num_train_samples}')
        logger.info(f'  Num Epochs = {num_train_epochs}')
        logger.info(f'  Instantaneous batch size per device = {batch_size}')
        logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
        logger.info(f'  Gradient Accumulation steps = {gradient_accumulation_steps}')
        logger.info(f'  Total optimization steps = {max_steps}')
        logger.info(
            f'  Number of trainable parameters = {get_num_trainable_parameters(text_encoder.get_input_embeddings())}'
        )

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        skip_first_batches = False

        if training_args.resume_from_checkpoint is not None:
            accelerator.load_state(training_args.resume_from_checkpoint)
            global_step = get_global_step(training_args.resume_from_checkpoint)
            epochs_trained = global_step // num_update_steps_per_epoch
            steps_trained_in_current_epoch = (global_step % num_update_steps_per_epoch) * gradient_accumulation_steps
            logger.info('  Continuing training from checkpoint, will skip to saved global_step')
            logger.info(f'  Continuing training from epoch {epochs_trained}')
            logger.info(f'  Continuing training from global step {global_step}')
            logger.info(
                f'  Will skip the first {epochs_trained} epochs then the first'
                f' {steps_trained_in_current_epoch} batches in the first epoch.'
            )

        progress_bar = tqdm(range(global_step, max_steps), desc='Steps', disable=not accelerator.is_local_main_process)

        for _ in range(epochs_trained, num_train_epochs):
            if skip_first_batches:
                train_dataloader = accelerate.skip_first_batches(train_dataloader, steps_trained_in_current_epoch)
                skip_first_batches = False

            text_encoder.train()
            for batch in train_dataloader:
                with accelerator.accumulate(text_encoder):
                    pixel_values = batch['pixel_values'].to(dtype=weight_dtype)
                    input_ids = batch['input_ids']
                    # Convert the images into latent space
                    latents = vae.encode(pixel_values).latent_dist.sample().detach()
                    latents = latents * 0.18215
                    # Sample and add noise to the latents
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (pixel_values.shape[0],), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    # Compute the text embedding
                    encoder_hidden_states = text_encoder(input_ids).last_hidden_state
                    encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)
                    # Predict the noise
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == 'epsilon':
                        target = noise
                    elif noise_scheduler.config.prediction_type == 'v_prediction':
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f'Unknown prediction type {noise_scheduler.config.prediction_type}')

                    loss = F.mse_loss(noise_pred, target, reduction='none').mean([1, 2, 3]).mean()
                    accelerator.backward(loss)

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(num_tokens) != placeholder_token_id
                    grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                    optimizer.step()
                    optimizer.zero_grad()
                
                metrics = {'loss': loss}

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                
                    maybe_log_or_save(
                        training_args, 
                        accelerator, 
                        global_step, 
                        metrics, 
                        progress_bar, 
                        num_update_steps_per_epoch
                    )

            accelerator.wait_for_everyone()

        # Log and save a final time
        maybe_log_or_save(
            training_args, 
            accelerator, 
            global_step, 
            metrics, 
            progress_bar, 
            num_update_steps_per_epoch,
            force_log=True,
            force_save=not our_training_args.do_not_force_save
        )
        if not our_training_args.do_not_save_pipeline:
            # Save a fine-tuned stable-diffusion pipeline
            save_pipeline(
                model_args,
                training_args,
                accelerator,
                text_encoder,
                vae,
                unet,
                tokenizer
            )

    _train_fn()


def load_prompts(inferece_args: InferenceArguments) -> List[str]:
    '''Load text prompts.

    Return the prompt or read prompts from a text file at the path
    provided by `inferece_args.prompt_or_path_to_prompts` or the
    `--prompt_or_path_to_prompts` command-line argument.

    Args:
        inference_args: A `InferenceArguments` object.

    Returns:
        A list of text prompts.
    '''
    if os.path.isfile(inferece_args.prompt_or_path_to_prompts):
        with open(inferece_args.prompt_or_path_to_prompts, 'r', encoding='utf-8') as f:
            return f.readlines()
    return [inferece_args.prompt_or_path_to_prompts]


def save_outputs(
    training_args: transformers.TrainingArguments,
    inference_args: InferenceArguments,
    output: diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput
) -> None:
    '''Save the stable-diffusion pipeline outputs.
    
    Args:
        training_args: A `transformers.TrainingArguments` object.
        inference_args: A `InferenceArguments` object.
        output: A stable-diffusion pipeline output.

    Raises:
        ValueError: The `output.images` format is unknown.
    '''
    for image in output.images:
        filename = f'{str(uuid.uuid4())}.png'
        path = [training_args.output_dir]
        if inference_args.inference_output_dir is not None:
            path.append(inference_args.inference_output_dir)
        path = os.path.join(*path)
        # Create the output directory if necessary
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)

        if isinstance(image, PIL.Image.Image):
            image.save(path)
        elif isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image, mode='RGB')
            image.save(path)
        else:
            raise ValueError(f'Saving images from a object of type `{type(image)} is not implemented!`')


def inference_fn(
    model_args: ModelArguments,
    data_args: DataArguments,
    inference_args: InferenceArguments,
    training_args: transformers.TrainingArguments
) -> None:
    accelerator = accelerate.Accelerator(
        mixed_precision=get_mixed_precision(training_args),
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=training_args.report_to,
        logging_dir=training_args.logging_dir
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    config = {
        **asdict(model_args),
        **asdict(data_args),
        **asdict(training_args)
    }
    config = convert_to_primitives(**config)
    logger.info(f'Config: \n{json.dumps(config, indent=2)}')

    if training_args.report_to is not None:
        accelerator.init_trackers(training_args.run_name, config)

    accelerate.utils.set_seed(training_args.seed)

    # Load the prompts from a text if necessary
    prompts = load_prompts(inference_args)

    pipeline = load_pipeline(
        inference_args,
        training_args, 
        accelerator
    )
    outputs = pipeline(
        prompts,
        inference_args.height,
        inference_args.width,
        inference_args.num_inference_steps,
        inference_args.guidance_scale,
        num_images_per_prompt=inference_args.num_images_per_prompt
    )

    save_outputs(training_args, inference_args, outputs)


def load_sweep_config(sweep_args: SweepArguments) -> Dict[str, Any]:
    '''Load the wandb sweep config from a JSON file.
    
    Args:
        sweep_args: A `SweepArguments` object.
    
    Returns:
        A dictionary of the JSON wandb sweep configuration.

    Raises:
        FileNotFoundError: The path specified with the `--sweep_config_path` arg.
    '''
    with open(sweep_args.sweep_config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def update_args(config: WandbConfig, args: Tuple[dataclass]) -> List[dataclass]:
    '''Update a dataclasses with settings from a wandb config.

    Args:
        config: The wandb config of type `wandb.sdk.wandb_config.Config`.
        args: A tuple of dataclasses.

    Returns:
        A list of the updated dataclass copies.
    '''
    updated = []
    for arg in args:
        tmp = copy.deepcopy(arg)
        for key, value in config.items():
            if key in dir(tmp):
                setattr(tmp, key, value)
        updated.append(tmp)
    return updated


def sweep_fn(
    model_args: ModelArguments,
    data_args: DataArguments,
    sweep_args: SweepArguments,
    our_training_args: OurTrainingArguments,
    training_args: transformers.TrainingArguments
) -> None:
    import wandb

    config = load_sweep_config(sweep_args)
    sweep_id = wandb.sweep(
        sweep=config, 
        project=training_args.run_name
    )
    logger.info(f'Sweeping with config: \n{json.dumps(config, indent=2)}')

    def _sweep_fn():
        run = wandb.init()
        
        # Update the args with the current sweep run config
        _model_args, _data_args, _training_args = \
            update_args(wandb.config, (model_args, data_args, training_args))
        _training_args.output_dir = os.path.join(
            _training_args.output_dir, f'{run.name}_{run.id}')

        train_fn(
            _model_args,
            _data_args,
            our_training_args,
            _training_args
        )

    wandb.agent(sweep_id, function=_sweep_fn, count=sweep_args.sweep_count)


def maybe_knockknock(knockknock_args: KnockKnockArguments) -> Callable:
    '''Return a decorator that does a `knockknock` setup.

    Args:
        knockknock_args: A `KnockKnockArguments``object.

    Returns:
        A function decorator for a `knockknock` setup based
        on the provided `KnockKnockArguments` object.
    '''
    def decorator(func: Callable) -> Callable:
        if knockknock_args.knockknock_on_discord:
            import knockknock
            
            # Read the webhook url from the args or environment variables
            webhook_url = knockknock_args.knockknock_discord_webhook_url
            if webhook_url is None:
                webhook_url = os.environ[KNOCKKNOCK_DISCORD_WEBHOOK_URL_ENV_VARIABLE]

            func = knockknock.discord_sender(webhook_url)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def main() -> None:
    parser = transformers.HfArgumentParser((
        ModelArguments, 
        DataArguments, 
        InferenceArguments,
        SweepArguments,
        KnockKnockArguments,
        OurTrainingArguments,
        transformers.TrainingArguments
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, inference_args, sweep_args, knockknock_args, our_training_args, training_args \
            = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, inference_args, sweep_args, knockknock_args, our_training_args, training_args \
            = parser.parse_args_into_dataclasses()

    @maybe_knockknock(knockknock_args)
    def _main():
        if sweep_args.do_sweep:
            sweep_fn(
                model_args, 
                data_args,
                sweep_args,
                our_training_args,
                training_args
            )
        else:
            if training_args.do_train:
                train_fn(
                    model_args, 
                    data_args,
                    our_training_args,
                    training_args
                )
            if inference_args.do_inference:
                inference_fn(
                    model_args, 
                    data_args, 
                    inference_args, 
                    training_args
                )

    _main()



if __name__ == '__main__':
    main()