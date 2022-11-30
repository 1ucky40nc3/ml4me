# Copyright 2022 Louis Wendler. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Parts of the following code are copied/adapted from:
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
# The original Apache License, Version 2.0 applies in theese cases.
# Thanks to the HuggingFace Inc. team for their generous contribution.
''' Finetune transformer models on the GLUE dataset '''

from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Callable,
    Optional
)

import os
import sys
import logging
from dataclasses import (
    dataclass,
    field
)

import numpy as np

import torch

import datasets
import evaluate
import transformers


logger = logging.getLogger(__name__)


Dataclass = Any
Dataset = Union[datasets.Dataset, datasets.DatasetDict]
Config = Any
Model = Any
Tokenizer = Any
DataCollator = Any


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2")
}


def required(help: str = '', **kwargs) -> Any:
    '''Return an field for a required arg.'''
    return field(
        metadata={
            'help': help
        }
    )


def argument(default: Any = None, help: str = '', **kwargs) -> Any:
    '''Return an field of given attributes.'''
    return field(
        default=default,
        metadata={
            'help': help
        }
    )


@dataclass
class DataArguments:
    '''Data arguments for the HfArgumentParser.'''
    task_name: Optional[str] = argument(
        help=(
            'Name of the GLUE task to train on.'
            f' Has to be one of {", ".join(TASK_TO_KEYS.keys())}'
        )
    )
    dataset_name: Optional[str] = argument(
        help='The name of the dataset to load from huggingface.co'
    )
    dataset_config_name: Optional[str] = argument(
        help='The name of configuration for the given dataset.'
    )
    max_seq_length: int = argument(
        default=128,
        help=(
            'The maximum number of input tokens.'
            ' Longer sequences will be truncated.'
        )
    )
    pad_to_max_length: bool = argument(
        default=True,
        help='State if input sequences shall all be padded to a max.'
    )
    max_train_samples: Optional[int] = argument(
        help='An optional number of max training samples (for debugging).'
    )
    max_eval_samples: Optional[int] = argument(
        help='An optional number of max evaluation samples (for debugging).'
    )
    max_predict_samples: Optional[int] = argument(
        help='An optional number of max test samples (for debugging).'
    )
    train_file: Optional[str] = argument(
        help='An optional CSV or JSON file with the training data.'
    )
    valid_file: Optional[str] = argument(
        help='An optional CSV or JSON file with the evaluation data.'
    )
    test_file: Optional[str] = argument(
        help='An optional CSV or JSON file with the test data.'
    )
    sentence1_key: Optional[str] = argument(
        help='A optional name for the first sentence.'
    )
    sentence2_key: Optional[str] = argument(
        help='A optional name for the second sentence.'
    )
    overwrite_cache: bool = argument(
        default=False,
        help='State if cached dataset files shall be overwritten.'
    )

    def __post_init__(self):
        if self.task_name:
            task_name = self.task_name.lower()
            if task_name not in TASK_TO_KEYS.keys():
                raise ValueError(
                    f'The given task {self.task_name}'
                    ' is not in the list of supported tasks'
                    f' {", ".join(TASK_TO_KEYS.keys())}'
                )
            self.task_name = task_name
        elif self.dataset_name:
            pass # Load from huggingface.co
        elif not self.train_file or not self.valid_file:
            raise ValueError(
                'A training/validation file needs to be specified, '
                ' if no `dataset_name` is given. Use a CSV/JSON format!'
            )
        else:
            _, train_extension = os.path.splitext(self.train_file)
            if train_extension not in ('.csv', '.json'):
                raise ValueError(
                    'The training file has to be in a CSV/JSON format!'
                )
            _, valid_extension = os.path.splitext(self.valid_file)
            if train_extension != valid_extension:
                raise ValueError(
                    'The training of validation files are required'
                    ' to have the same format. Use either CSV or JSON.'
                )
            if self.test_file:
                _, test_extension = os.path.splitext(self.test_file)
                if train_extension != test_extension:
                    raise ValueError(
                        'The training of test files are required'
                        ' to have the same format. Use either CSV or JSON.'
                    )
            
            if not self.sentence1_key or not self.sentence2_key:
                raise ValueError(
                    'The `sentence1_key` and `sentence2_key` args'
                    ' have to be given when data is loaded from CSV/JSON.'
                )


@dataclass
class ModelArguments:
    '''Model arguments for the HfArgumentParser'''
    model_name_or_path: str = required(
        help=(
            'The name or path of a model.'
            ' If no path is given the model is load from huggingface.co'
        )
    )
    config_name: Optional[str] = argument(
        help="A model's config as model name or path"
    )
    tokenizer_name: Optional[str] = argument(
        help='Name of path of a pretrained tokenizer.'
    )
    use_fast_tokenizer: bool = argument(
        default=True,
        help='State if a fast version of a pretrained tokenizer shall be used.'
    )
    cache_dir: Optional[str] = argument(
        help=(
            'Cache location for models loaded from huggingface.co.'
            ' By default this path is "~/.cache/huggingface".'
        )
    )
    revision: str = argument(
        default='main',
        help=(
            'A model version that shall be used.'
            ' This can be either a branch/tag/commit name/id.'
        )
    )
    use_auth_token: Optional[bool] = argument(
        default=None,
        help=(
            'State if a login with a auth token shall be tried.'
            ' The auth token is automatically generated via the'
            ' `huggingface-cli login` command outside this script.'
            ' This is needed for the use of private models or data.'
        )
    )
    ignore_mismatched_sizes: bool = argument(
        default=False,
        help='State if mismatched model head dimemsions shall be ignored.'
    )


def setup_logging(args: Dataclass) -> None:
    '''Set up the logging.'''
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()


def display_summary(args: Dataclass) -> None:
    '''Display a summary of the training arguments.'''
    logger.warning(
        f'Process rank: {args.local_rank},'
        f' device: {args.device},'
        f' n_gpu: {args.n_gpu},'
        f' distributed training: {bool(args.local_rank != -1)},'
        f' 16-bits training: {args.fp16}'
    )
    logger.info(f'Training/evaluation args: {args}')


def detect_last_checkpoint(args: Dataclass) -> Optional[str]:
    '''Detect the last saved checkpoint from previous runs.'''
    last_checkpoint = None

    if (
        os.path.isdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(args.output_dir)
        if not last_checkpoint and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f'The `output_dir` is not empty.'
                ' Restart with the arg `--overwrite_output_dir`!'
            )
        elif last_checkpoint and not args.resume_from_checkpoint:
            logger.info(f'Training will resume from checkpoint {last_checkpoint}.')

    return last_checkpoint


def load_raw_datasets(
    data_args: Dataclass, 
    model_args: Dataclass,
    train_args: Dataclass,
) -> Dataset:
    '''Load a raw dataset from huggingface.co or a CSV/JSON file.'''
    kwargs = dict(
        cache_dir=model_args.cache_dir,
        use_auth_token=model_args.use_auth_token
    )

    if data_args.task_name:
        raw_datasets = datasets.load_dataset(
            'glue',
            data_args.task_name,
            **kwargs
        )
    elif data_args.dataset_name:
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            **kwargs
        )
    else:
        data_files = dict(
            train=data_args.train_file,
            validation=data_args.valid_file
        )
        if train_args.do_predict:
            if not data_args.test_file:
                raise ValueError('No `test_file` specified!')
            else:
                data_files['test'] = data_files.test_file
        
        _, extension = os.path.splitext(data_args.train_file)
        extension = extension[1:]

        raw_datasets = datasets.load_dataset(
            extension,
            data_files=data_files,
            **kwargs
        )
    
    return raw_datasets


def get_dataset_info(
    dataset: Dataset,
    data_args: Dataclass
) -> Tuple[bool, int, List[int]]:
    '''Return if we do regression, the number of labels and unique labels.'''
    split = dataset['train']
    label = split.features['label']
    if data_args.task_name:
        is_regression = data_args.task_name == 'stsb'
        if not is_regression:
            label_list = label.names
            num_labels = len(label_list)
        else:
            label_list = []
            num_labels = 1
    else:
        is_regression = label.dtype in ['float32', 'float64']
        if is_regression:
            label_list = []
            num_labels = 1
        else:
            label_list = split.unique('label')
            label_list.sort()
            num_labels = len(label_list)
    
    return is_regression, num_labels, label_list


def evaluate_max(max_length: Optional[int], dataset: Dataset) -> int:
    '''Evaluate if the value or the dataset length shall be returned'''
    if not max_length:
        return len(dataset)
    return min(max_length, len(dataset))


def preprocess_datasets(
    dataset: Dataset, 
    data_args: Dataclass,
    train_args: Dataclass,
    is_regression: bool,
    num_labels: int,
    label_list: List[int],
    config: Config,
    model: Model,
    tokenizer: Tokenizer
) -> Tuple[Optional[Dataset], List[Dataset], List[Dataset]]:
    '''Return the preprocessed training, evaluation and test datasets.'''
    train_dataset = None
    eval_datasets = []
    test_datasets = []

    if data_args.task_name:
        sentence1_key, sentence2_key = TASK_TO_KEYS[data_args.task_name]
    else:
        sentence1_key = data_args.sentence1_key
        sentence2_key = data_args.sentence2_key

    if data_args.pad_to_max_length:
        padding = 'max_length'
    else:
        padding = False

    # Set the order of labels to use
    label_to_id = None
    label2id = transformers.PretrainedConfig(num_labels=num_labels).label2id
    if (
        model.config.label2id != label2id
        and data_args.task_name
        and not is_regression
    ):
        label_name_to_id = {
            k.lower(): v 
            for k, v in model.config.label2id.items()
        }
        label_names = list(sorted(label_name_to_id.keys()))
        sorted_label_list = list(sorted(label_list))
        if label_names == sorted_label_list:
            label_to_id = {
                i: int(label_name_to_id[label_list[i]])
                for i in range(num_labels)
            }
        else:
            logger.warning(
                'Model pretrained with mismatched labels.'
                f' The pretrained labels {label_names}'
                f" don't match with the labels {sorted_label_list}."
                f' As a consequence the new labels will be ignored.'
            )
    elif not data_args.task_name and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    
    if label_to_id:
        model.config.label2id = label_to_id
        model.config.id2label = {v: k for k, v in config.label2id.items()}
    elif data_args.task_name and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {v: k for k, v in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f'The `max_seq_length` arg is to big for the model'
            f' the arg will be restricted to the value `{tokenizer.max_seq_length}`.'
        )
    max_seq_length = min(
        data_args.max_seq_length,
        tokenizer.model_max_length
    )

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if not sentence2_key else
            (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args,
            padding=padding,
            max_length=max_seq_length,
            truncation=True
        )
        if label_to_id and 'label' in examples:
            result['label'] = [
                (label_to_id[l] if l != -1 else -1)
                for l in examples['label']
            ]
        return result
    
    with train_args.main_process_first(desc='dataset map pre-processing'):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc='Running tokenizer on dataset'
        )

    if train_args.do_train:
        if 'train' not in dataset:
            raise ValueError("The provided dataset doesn't have a train split!")

        train_dataset = dataset['train']
        if data_args.max_train_samples:
            max_train_samples = evaluate_max(data_args.max_train_samples, train_dataset)
            train_dataset = train_dataset.select(range(max_train_samples))
    
    if train_args.do_eval:
        for key in ['validation', 'validation_matched']:
            if key in dataset:
                eval_dataset = dataset[key]
                if data_args.max_eval_samples is not None:
                    max_eval_samples = evaluate_max(data_args.max_eval_samples, eval_dataset)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                
                eval_datasets.append(eval_dataset)
        
        if len(eval_datasets) == 0:
            raise ValueError("The provided dataset doesn't have a validation split!")

    if (
        train_args.do_predict
        or data_args.task_name
        or data_args.test_file
    ):
        for key in ['test', 'test_matched']:
            if key in dataset:
                test_dataset = dataset[key]
                if data_args.max_predict_samples is not None:
                    max_predict_samples = evaluate_max(data_args.max_predict_samples, test_dataset)
                    test_dataset = test_dataset.select(range(max_predict_samples))
                
                test_datasets.append(test_dataset)
        
        if len(test_datasets) == 0:
            raise ValueError("The provided dataset doesn't have a test split!")

    return train_dataset, eval_datasets, test_datasets


def interpret_predictions(
    prediction: torch.Tensor,
    is_regression: bool
) -> torch.Tensor:
    '''Interpret a prediction based on the task nature.'''
    if is_regression:
        return np.squeeze(prediction)
    return np.argmax(prediction, axis=1)


def load_metrics(
    data_args: Dataclass,
    is_regression: bool
) -> Callable[[transformers.EvalPrediction], Dict[str, Any]]:
    '''Load a metric and return a `compute_metrics` function.'''
    if data_args.task_name:
        metric = evaluate.load('glue', data_args.task_name)
    elif is_regression:
        metric = evaluate.load('mse')
    else:
        metric = evaluate.load('accuracy')
    
    def compute_metrics(
        eval_prediction: transformers.EvalPrediction
    ) -> Dict[str, Any]:
        '''A function which computes metrics.'''
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = interpret_predictions(predictions, is_regression)
        
        return metric.compute(
            predictions=predictions, 
            references=eval_prediction.label_ids
        )

    return compute_metrics


def load_collator(
    data_args: Dataclass,
    train_args: Dataclass, 
    tokenizer: Tokenizer
) -> Optional[DataCollator]:
    '''Return a data collator.'''
    data_collator = None
    
    if data_args.pad_to_max_length:
        data_collator = transformers.default_data_collator
    elif train_args.fp16:
        data_collator = transformers.DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=8
        )
    
    return data_collator


def main():
    parser = transformers.HfArgumentParser((
        DataArguments,
        ModelArguments,
        transformers.TrainingArguments
    ))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()
    
    setup_logging(train_args)
    display_summary(train_args)

    # Set seed before model init
    transformers.set_seed(train_args.seed)

    raw_datasets = load_raw_datasets(
        data_args,
        model_args,
        train_args
    )
    is_regression, num_labels, label_list = get_dataset_info(
        raw_datasets,
        data_args 
    )

    config = transformers.AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.revision,
        use_auth_token=model_args.use_auth_token,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        cache_dir=model_args.cache_dir,
        revision=model_args.revision,
        use_auth_token=model_args.use_auth_token,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool('.ckpt' in model_args.model_name_or_path),
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        cache_dir=model_args.cache_dir,
        revision=model_args.revision,
        use_auth_token=model_args.use_auth_token,
    )

    train_dataset, eval_datasets, predict_datasets = preprocess_datasets(
        raw_datasets,
        data_args,
        train_args,
        is_regression,
        num_labels,
        label_list,
        config,
        model,
        tokenizer
    )
    data_collator = load_collator(
        data_args,
        train_args,
        tokenizer
    )

    compute_metrics = load_metrics(data_args, is_regression)

    trainer = transformers.Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets[0],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    if train_args.do_train:
        resume_from_checkpoint = detect_last_checkpoint(train_args)
        if train_args.resume_from_checkpoint:
            resume_from_checkpoint = train_args.resume_from_checkpoint

        # Train!
        train_result = trainer.train(resume_from_checkpoint)
        
        metrics = train_result.metrics
        max_train_samples = evaluate_max(data_args.max_train_samples, train_dataset)
        metrics['train_samples'] = max_train_samples

        # Save the model and the tokenizer.
        trainer.save_model()

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
    
    if train_args.do_eval:
        logger.info('*** Evaluate ***')

        combined = {}

        for i, eval_dataset in enumerate(eval_datasets):
            metrics = trainer.evaluate(eval_dataset)

            max_eval_samples = evaluate_max(data_args.max_eval_samples, eval_dataset)
            metrics['eval_samples'] = max_eval_samples

            if i > 0:
                # Only the mlni dataset has two eval datasets.(validation & validation_mismatched)
                # Mark the metrics of the validation_mismatched dataset with the `_mm` suffix.
                metrics = {f'{k}_mm': v for k, v in metrics.items()}
            combined.update(metrics)

            trainer.log_metrics('eval', combined)
            trainer.save_metrics('eval', combined)

    if train_args.do_predict:
        logger.info('*** Predict ***')

        for predict_dataset in predict_datasets:
            predict_dataset = predict_dataset.remove_columns('label')
            predict_output = trainer.predict(
                predict_dataset,
                metric_key_prefix='predict'
            )
            predictions = predict_output.predictions
            predictions = interpret_predictions(predictions, is_regression)
            
            # Only the MNLI dataset has two tasks
            task_name = data_args.task_name if i < 1 else 'mnli-mm'
            output_predict_file = os.path.join(
                train_args.output_dir,
                f'predict_results_{task_name}.csv'
            )
            if trainer.is_world_process_zero():
                df = predict_dataset.to_pandas()
                # Remove an unnamed column that is automatically generated
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

                # Map the predictions with the label list
                predictions = predictions.tolist()
                id_to_label = lambda i: label_list[i]
                predictions = map(id_to_label, predictions)
                predictions = list(predictions)
                df['preds'] = predictions

                df.to_csv(output_predict_file, sep='\t')
    
    kwargs = {
        'tasks': 'text-classification',
        'finetuned_from': model_args.model_name_or_path
    }
    if data_args.task_name:
        kwargs['language'] = 'en'
        kwargs['dataset_tags'] = 'glue'
        kwargs['dataset_args'] = data_args.task_name
        kwargs['dataset'] = f'GLUE {data_args.task_name.upper}'

    if train_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == '__main__':
    main()

