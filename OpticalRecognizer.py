import pytesseract
import numpy as np
from pytesseract import Output

class OpticalRecognizer:

    language: str

    def __init__(self, lang: str):
        self.language = lang

    def setLanguage(self, lang: str) -> None:
        self.language = lang

    def getLanguage(self) -> str:
        return self.language

    def im2text(self, img: np.ndarray, code: int = 0) -> str:
        """ This function takes an image and applies an OCR algorithm to convert the
           text contained in the image into a string.

        Args:
            img (np.ndarray): image pixels matrix.
            code (int, optional): Code defining how the text is accommodated
                                  in the image. This facilitates the work to
                                  the OCR algorithm and increase the precision.
                                  Defaults to 0.
                                - 0 : Single text line.
                                - 1 : Single word.
                                - 2 : Single character.

        Returns:
            str: text contained in the image.
        """
        # Single text line
        if (code == 0):
            config = "--psm 7"
        # Single word
        elif (code == 2):
            config = "--psm 8"
        # Single character
        else:
            config = "--psm 10"

        return pytesseract.image_to_string(image=img,
                                           lang=self.language,
                                           config=config).replace("\n\x0c", "")
