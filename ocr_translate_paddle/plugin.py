###################################################################################
# ocr_translate-hugging_face - a plugin for ocr_translate                         #
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
# Home: https://github.com/Crivella/ocr_translate-hugging_face                    #
###################################################################################
"""ocr_translate plugin to allow loading of PaddleOCR."""
import logging
import os

import numpy as np
from ocr_translate import models as m
from paddleocr import PaddleOCR
from PIL import Image

logger = logging.getLogger('plugin')

class PaddleOCRModel(m.OCRModel):
    """OCRtranslate plugin to allow loading of PaddleOCR as text OCR."""
    class Meta: # pylint: disable=missing-class-docstring
        proxy = True

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)

        self.reader = None
        self.dev = os.environ.get('DEVICE', 'cpu')

    def load(self):
        """Load the model into memory."""
        logger.debug('Loading PaddleOCR model (nothing to do here)')

    def unload(self) -> None:
        """Unload the model from memory."""
        logger.debug('Unloading PaddleOCR model (nothing to do here)')

    def _ocr(
            self,
            img: Image.Image, lang: str = None, options: dict = None
            ) -> str:
        """Perform OCR on an image.

        Args:
            img (Image.Image):  A Pillow image on which to perform OCR.
            lang (str, optional): The language to use for OCR. (Not every model will use this)
            bbox (tuple[int, int, int, int], optional): The bounding box of the text on the image in lbrt format.
            options (dict, optional): A dictionary of options to pass to the OCR model.

        Raises:
            TypeError: If img is not a Pillow image.

        Returns:
            str: The text extracted from the image.
        """

        ocr = PaddleOCR(
            use_angle_cls=True, lang=lang,
            use_gpu=(self.dev == 'cuda')
            )
        result = ocr.ocr(np.array(img), cls=True)[0]

        logger.debug(f'PaddleOCR result: {result}')

        generated_text = []
        for line in result:
            # box = np.array(line[0])
            # l = box[:, 0].min()
            # t = box[:, 1].min()
            # r = box[:, 0].max()
            # b = box[:, 1].max()
            text, _ = line[1]
            generated_text.append(text)

        generated_text = ' '.join(generated_text)

        return generated_text
