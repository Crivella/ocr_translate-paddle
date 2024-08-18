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

import pytest

from ocr_translate_paddle import plugin as paddle


def test_env_none(monkeypatch):
    """Test that no env set causes ValueError."""
    monkeypatch.delenv('OCT_BASE_DIR', raising=False)
    with pytest.raises(ValueError):
        paddle.EnvMixin()

def test_env_paddleocr_prefix(prefix):
    """Test that the PADDLEOCR_PREFIX environment variable is set."""
    assert not prefix.exists()
    cls = paddle.EnvMixin()
    assert cls.basedir == prefix
    assert prefix.exists()

def test_env_base_dir(base):
    """Test that the OCT_BASE_DIR environment variable is set."""
    assert not base.exists()
    cls = paddle.EnvMixin()
    assert str(cls.basedir).startswith(str(base))
    assert base.exists()

def test_env_device(device):
    """Test that the DEVICE environment variable is set."""
    cls = paddle.EnvMixin()
    assert cls.dev == device

def test_env_device_cuda(device_cuda):
    """Test that the DEVICE environment variable is set."""
    cls = paddle.EnvMixin()
    assert cls.dev == device_cuda
