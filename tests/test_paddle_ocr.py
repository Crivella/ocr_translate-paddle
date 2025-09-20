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
"""Tests for paddle plugin."""

# pylint: disable=protected-access
from ocr_translate_paddle import plugin as paddle


def test_load_ocr_model_paddle(monkeypatch, paddleocr_ocr_model):
    """Test load box model. Success"""
    monkeypatch.setattr(paddle, 'PaddleOCR', lambda *args, **kwargs: 'mocked')
    monkeypatch.setattr(paddleocr_ocr_model, 'DISABLE_LOAD_EVENTS', True)
    paddleocr_ocr_model.load()
    assert paddleocr_ocr_model.reader is None

def test_unload_ocr_model_paddle(paddleocr_ocr_model):
    """Test unload box model. Success"""
    paddleocr_ocr_model.lang = 'en'
    paddleocr_ocr_model.reader = 'mocked'

    paddleocr_ocr_model.unload()

    assert paddleocr_ocr_model.lang is None
    assert paddleocr_ocr_model.reader is None

def test_paddle_once(monkeypatch, paddleocr_ocr_model, image_pillow, mock_reader_ocr):
    """Test paddle ocr once. Success"""
    monkeypatch.setattr(paddle, 'PaddleOCR', lambda *args, **kwargs: mock_reader_ocr())

    res = paddleocr_ocr_model._ocr(image_pillow, lang='en')

    assert res == 'mock_line1 mock_line2'

def test_paddle_ocr_twice_samelang(monkeypatch, paddleocr_ocr_model, image_pillow, mock_reader_ocr):
    """Test paddle ocr once. Success"""
    monkeypatch.setattr(paddle, 'PaddleOCR', lambda *args, **kwargs: mock_reader_ocr())

    paddleocr_ocr_model._ocr(image_pillow, lang='en')
    reader1 = paddleocr_ocr_model.reader
    paddleocr_ocr_model._ocr(image_pillow, lang='en')
    reader2 = paddleocr_ocr_model.reader

    assert reader1 is reader2
    assert paddleocr_ocr_model.lang == 'en'

def test_paddle_ocr_twice_difflang(monkeypatch, paddleocr_ocr_model, image_pillow, mock_reader_ocr):
    """Test paddle ocr once. Success"""
    monkeypatch.setattr(paddle, 'PaddleOCR', lambda *args, **kwargs: mock_reader_ocr())

    paddleocr_ocr_model._ocr(image_pillow, lang='en')
    reader1 = paddleocr_ocr_model.reader
    assert paddleocr_ocr_model.lang == 'en'
    paddleocr_ocr_model._ocr(image_pillow, lang='ch')
    reader2 = paddleocr_ocr_model.reader
    assert paddleocr_ocr_model.lang == 'ch'

    assert reader1 is not reader2
