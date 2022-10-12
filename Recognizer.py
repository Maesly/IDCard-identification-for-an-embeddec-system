from DataBase import DataBase
from OpticalRecognizer import OpticalRecognizer
from ImageProcessor import ImageProcessor
from ImageCapture import ImageCapture

class Recognizer:

    database: DataBase
    optRecognizer: OpticalRecognizer
    imgProcessor: ImageProcessor

    def __init__(self, database: DataBase, lang: str):
        self.database = database
        self.database.loadDatabase()
        self.optRecognizer = OpticalRecognizer(lang)
        self.imgProcessor = ImageProcessor()

    def getDatabase(self) -> DataBase:
        return self.database

    def getLanguage(self) -> str:
        return self.optRecognizer.getLanguage()

    def getOptRecognizer(self) -> OpticalRecognizer:
        return self.optRecognizer

    def getImgProcessor(self) -> ImageProcessor:
        return self.imgProcessor

    def setDatabase(self, database: DataBase) -> None:
        self.database = database

    def setLanguage(self, lang: str) -> None:
        self.optRecognizer.setLanguage(lang)

    def setOptRecognizer(self, optRecognizer: OpticalRecognizer) -> None:
        self.optRecognizer = optRecognizer

    def setImgProcessor(self, imgProcessor: ImageProcessor) -> None:
        self.imgProcessor = imgProcessor
