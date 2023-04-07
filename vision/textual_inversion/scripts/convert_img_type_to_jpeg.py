#!/usr/bin/env python
# coding=utf-8
# MIT License
# 
# Copyright (c) 2023 Louis Wendler
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
'''A script to convert images from the ".heic" format to ".jpeg".

Typical usage example:

>>> python3 convert_heic_to_jpeg.py --input_dir ./heic --output_dir ./jpeg
'''


from typing import List

import os
import glob

from absl import (
    app,
    flags
)

import pyheif
from PIL import Image
from tqdm import tqdm


FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='input_dir',
    default=None,
    help='The directory containing images in the ".heic" format.'
)
flags.DEFINE_string(
    name='output_dir',
    default=None,
    help='The output directory fpr ".jpeg" files.'
)
flags.DEFINE_string(
    name='pattern',
    default='*.heic',
    help=(
        'A pattern to match files in the input directory.'
        ' We match files with a ".heic" extension by default.'
    ) 
)

flags.mark_flags_as_required([
    'input_dir',
    'output_dir'
])


def ls_dir(directory: str, pattern: str = '*.heic') -> List[str]:
    '''List the contents of a directory.

    Args:
        directory: The path to a directory.
        pattern: The format filter for directory contents.
                 This could be a regular expression to match filenames or types.

    Returns:
        A list of matched paths in the directory.
    '''
    pathname = os.path.join(directory, pattern)
    paths = glob.glob(pathname)
    return paths


def get_filename(path: str) -> str:
    '''Return the filename from a file's path.

    Args:
        path: The file's path.

    Returns:
        The filename of the file.

    Raises:
        ValueError: The `path` is not linked to a file.
    '''
    if not os.path.isfile(path):
        raise ValueError(f'The given path is not linked to a file! Path: {path}')
    path, _ = os.path.splitext(path)
    filename = os.path.basename(path)
    return filename


def get_ext(path: str) -> str:
    '''Return the extension from a file's path.
    
    Args:
        path: The file's path.

    Returns:
        The extension of the file.

    Raises:
        ValueError: The `path` is not linked to a file.
    '''
    if not os.path.isfile(path):
        raise ValueError(f'The given path is not linked to a file! Path: {path}')
    _, ext = os.path.splitext(path)
    return ext


def main(argv):
    del argv

    paths = ls_dir(FLAGS.input_dir, pattern=FLAGS.pattern)
    pbar = tqdm(paths, desc='Converting images')
    for path in pbar:
        # Load the image data
        ext = get_ext(path)        
        if ext.lower() == '.heic':
            heif_file = pyheif.read(path)
            pil_img = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                'raw',
                heif_file.mode,
                heif_file.mode
            )
        else:
            pil_img = Image.open(path).convert('RGB')
        
        # Convert & save the image
        filename = get_filename(path)
        path = os.path.join(FLAGS.output_dir, f'{filename}.jpeg')
        pil_img.save(path, format='jpeg')
        


if __name__ == '__main__':
    app.run(main)
    