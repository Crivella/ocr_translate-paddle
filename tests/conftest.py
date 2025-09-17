###################################################################################
# ocr_translate-paddle - a plugin for ocr_translate                               #
# Copyright (C) 2023-present Davide Grassano                                      #
#                                                                                 #
# This program is free software: you can redistribute it and/or modify            #
# it under the terms of the GNU General Public License as published by            #
# the Free Software Foundation, either version 3 of the License.                  #
#                                                                                 #
# This program is distributed in the hope that it will be useful,                 #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                  #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                   #
# GNU General Public License for more details.                                    #
#                                                                                 #
# You should have received a copy of the GNU General Public License               #
# along with this program.  If not, see {http://www.gnu.org/licenses/}.           #
#                                                                                 #
# Home: https://github.com/Crivella/ocr_translate-paddle                          #
###################################################################################
"""Fixtures for tests."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ocr_translate_paddle import plugin as paddle

strings = [
    'This is a test string.',
    'This is a test string.\nWith a newline.',
    'This is a test string.\nWith a newline.\nAnd another.',
    'This is a test string.? With a special break character.',
    'This is a test string.? With a special break character.\nAnd a newline.',
    'String with a dash-newline brok-\nen word.'
]
ids = [
    'simple',
    'newline',
    'newlines',
    'breakchar',
    'breakchar_newline',
    'dash_newline'
]

@pytest.fixture(autouse=True)
def base(monkeypatch, tmpdir) -> Path:
    """Mock base classes."""
    tmp = str(tmpdir / 'base')
    monkeypatch.setenv('OCT_BASE_DIR', tmp)
    return Path(tmp)

@pytest.fixture()
def prefix(monkeypatch, tmpdir) -> Path:
    """Mock base classes."""
    tmp = str(tmpdir / 'prefix')
    monkeypatch.setenv('PADDLEOCR_PREFIX', tmp)
    return Path(tmp)

@pytest.fixture()
def device(monkeypatch) -> str:
    """Mock device."""
    monkeypatch.setenv('DEVICE', 'cpu')
    return 'cpu'

@pytest.fixture()
def device_cuda(monkeypatch) -> str:
    """Mock device."""
    monkeypatch.setenv('DEVICE', 'cuda')
    return 'cuda'

@pytest.fixture(params=strings, ids=ids)
def string(request):
    """String to perform TSL on."""
    return request.param

@pytest.fixture()
def batch_string(string):
    """Batched string to perform TSL on."""
    return [string, string, string]

@pytest.fixture(scope='session')
def image_pillow():
    """Random Pillow image."""
    npimg = np.random.randint(0,255,(25,25,3), dtype=np.uint8)
    return Image.fromarray(npimg)

# @pytest.fixture()
# def paddleocr_box_model() -> paddle.PaddleBOXModel:
#     """OCRBoxModel database object."""
#     paddle_model_dict = {
#         'name': 'paddleocr',
#         'language_format': 'iso1',
#         'entrypoint': 'paddle.box'
#     }

#     return paddle.PaddleBOXModel(**paddle_model_dict)

@pytest.fixture()
def paddleocr_ocr_model() -> paddle.PaddleOCRModel:
    """OCRBoxModel database object."""
    paddle_model_dict = {
        'name': 'paddleocr',
        'language_format': 'iso1',
        'entrypoint': 'paddle.ocr'
    }

    return paddle.PaddleOCRModel(**paddle_model_dict)

@pytest.fixture(scope='function')
def mock_called(request):
    """Generic mock function to check if it was called."""
    def mock_call(*args, **kwargs): # pylint: disable=inconsistent-return-statements
        mock_call.called = True
        mock_call.args = args
        mock_call.kwargs = kwargs

        if hasattr(request, 'param'):
            return request.param

    if hasattr(request, 'param'):
        mock_call.expected = request.param

    return mock_call

# RESULTS #0 --------------
# |  input_path: None
# |  page_index: None
# |  doc_preprocessor_res:
# |  |  input_path: None
# |  |  page_index: None
# |  |  input_img: array<(182, 28, 3)>
# |  |  model_settings:
# |  |  |  use_doc_orientation_classify: True
# |  |  |  use_doc_unwarping: True
# |  |  angle: 0
# |  |  rot_img: array<(182, 28, 3)>
# |  |  output_img: array<(182, 28, 3)>
# |  dt_polys: [array([[  0,   9],
#        ...,
#        [  0, 182]]
# |  model_settings:
# |  |  use_doc_preprocessor: True
# |  |  use_textline_orientation: True
# |  text_det_params:
# |  |  limit_side_len: 64
# |  |  limit_type: min
# |  |  thresh: 0.3
# |  |  max_side_limit: 4000
# |  |  box_thresh: 0.6
# |  |  unclip_ratio: 1.5
# |  text_type: general
# |  text_rec_score_thresh: 0.0
# |  return_word_box: False
# |  rec_texts: ['トレトよね']
# |  rec_scores: [0.9329773187637329]
# |  rec_polys: [array([[  0,   9],
#        ...,
#        [  0, 182]]
# |  vis_fonts: [<paddlex.utils.fonts.Font object at 0x7eb6043e2d5
# |  textline_orientation_angles: [0]
# |  rec_boxes: array<(1, 4)>

@pytest.fixture(scope='function')
def mock_reader_ocr():
    """Mock paddleocr reader."""
    class MockReader(): # pylint: disable=missing-class-docstring
        res = [
            # [
                # [((1,3), (1,4), (2,3), (2,4)), ('mock_line1', 1.0)],
                # [((5,7), (5,8), (6,7), (6,8)), ('mock_line2', 1.0)],
            {
                'rec_texts': ['mock_line1', 'mock_line2'],
                'rec_scores': [1.0, 1.0]
            },
            # {'rec_text': ['mock_line2'], 'rex_scores': [1.0]}
            # ]
        ]
        def __init__(self):
            self.called = False

        def predict(self, *args, **kwargs): # pylint: disable=missing-function-docstring,unused-argument
            self.called = True
            return MockReader.res

    return MockReader
