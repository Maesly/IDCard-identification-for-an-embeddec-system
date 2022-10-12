from ImageProcessor import ImageProcessor
from Recognizer import Recognizer
from CRElectoralRoll import CRElectoralRoll
import numpy as np
import cv2
import math
import re
import matplotlib.pyplot as plt
import random as rng
from PIL import Image

class CRIdentityCardRecognizer(Recognizer):

    resizedSize: int
    idNumberX1: int
    idNumberX2: int
    idNumberY1: int
    idNumberY2: int
    idNameX1: int
    idNameX2: int
    idNameY1: int
    idNameY2: int
    segMinSize: int
    padding: int
    rectKernel: np.ndarray
    sqKernel3: np.ndarray
    sqKernel7: np.ndarray

    def __init__(self):
        super().__init__(database=CRElectoralRoll(), lang="spa")
        self.idNumberX1 = 120
        self.idNumberX2 = 335
        self.idNumberY1 = 75
        self.idNumberY2 = 128
        self.idNameX1 = 150
        self.idNameX2 = 380
        self.idNameY1 = 230
        self.idNameY2 = 300
        self.segMinSize = 7
        self.padding = 3
        self.resizedSize = 500
        self.nameSize = 16
        self.rectKernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT,
                                                    ksize=(15, 15))
        self.sqKernel3 = cv2.getStructuringElement(shape=cv2.MORPH_RECT,
                                                   ksize=(3, 3))
        self.sqKernel7 = cv2.getStructuringElement(shape=cv2.MORPH_RECT,
                                                   ksize=(7, 7))

    def preprocesImage(self, filePath: str):
        '''
        Function that apply all the process to enhance the image
        Args: 
            filePath (str): image file path. 
        '''
        # Open the image taken
        foto = self.imgProcessor.readImage(filePath=filePath)
        
        # Create a copy to cut the image
        copia = np.copy(foto)
        
        # Apply gray scale
        gray = self.imgProcessor.im2grayscale(foto)   
        cv2.imwrite('gray.jpg', gray)
        # Apply ecualization
        #ecua = self.imgProcessor.histogram(gray)
        #cv2.imwrite('ecua.jpg', ecua)
        # Apply threshold
        threshold = self.imgProcessor.apply_adaptive_threshold(filePath=gray)
        cv2.imwrite('threshold.jpg', threshold)
        canny = self.imgProcessor.canny(threshold)
        cv2.imwrite('canny.jpg', canny)
        #find contours        
        contours = self.imgProcessor.countoursImage(threshold)
        draw = self.imgProcessor.drawContours(gray, contours)
        cv2.imwrite('draw.jpg', draw)
        # Cutting the edges of the image
        cut = self.imgProcessor.cutImage(copia, contours=contours)
        
        # Sharpen the image
        
        sharp = self.imgProcessor.sharpenImagen(cut)
    
        # Save the image
        cv2.imwrite(filePath, sharp)

        return

    #def validateIDCard(self, filePath: str):

    def verifyIdCardAuthenticity(self, filePath: str, verbose: bool = False) -> bool:
        """ This function takes an input image of an identification card and verifies
            its authenticity.
        Args:
            filePath (str): image file path.
            verbose (bool, optional): flag to show the predictions and the
                                      segmented image. Defaults to False.
        Returns:
            bool: identification authenticity.
        """
        # Reads and loads the input image
        img = self.imgProcessor.readImage(filePath=filePath)
        # Resizes the image into a defined size
        imgResized = self.imgProcessor.imResize(img=img,
                                                width=self.resizedSize)


        # Converts the color image into a grayscale image
        gray = self.imgProcessor.im2grayscale(img=imgResized)
        # Segments the image
        imgSegmented = self.segmentImage(img=gray,
                                         k1=self.rectKernel,
                                         k2=self.sqKernel3)
        # Detects the contours of the segments
        contours = self.imgProcessor.findContours(img=imgSegmented)
        # Gets the segment corresponding to the identification number
        id, idSegment = self.getId(img=gray, contours=contours)
        
        doc, docSegment = self.getDocumentType(img=gray, contours=contours)
        #print(idSegment)
        # Gets the segment corresponding to the full name
        fullName, fullNameSegment = self.getFullName(img=gray,
                                                     contours=contours)
        # Determines if the information is authentic
        isAuthentic = self.database.isAuthentic(numId=id.replace(" ", ""),
                                                nameId=fullName)
        # If verbose is activated shows some process information
        if (verbose):
            # Shows the prediction information
            self.showInformation(id=id,
                                 fullName=fullName,
                                 authentic=isAuthentic)
            # Shows the image with its target segments
            cv2.imshow("Image", self.drawSegments(img=gray,
                                                  segments=[idSegment, fullNameSegment]))

        return isAuthentic

    def drawSegments(self, img: np.ndarray, segments: list) -> np.ndarray:
        """ This function takes an image and draws all the segments contained
            in the given list.

        Args:
            img (np.ndarray): input image.
            segments (list): segments list to draw.

        Returns:
            np.ndarray: output image.
        """
        # Goes through the all the segments
        for i in range(len(segments)):
            if (segments[i]):
                img = self.imgProcessor.drawSegment(img=img,
                                                    segment=segments[i])

        return img

    def showInformation(self, id: int, fullName: str, authentic: bool) -> None:
        """ This function shows the given information in a formatted way.

        Args:
            id (int): identification number.
            fullName (str): identification full name.
            authentic (bool): identification authenticity.
        """
        if (fullName):
            name = " ".join(fullName.split()[:-2])
            firstLastName = fullName.split()[-2]
            secondLastName = fullName.split()[-1]
        else:
            name = None
            firstLastName = None
            secondLastName = None

        print("id               : {}".format(id))
        print("Name             : {}".format(name))
        print("First last name  : {}".format(firstLastName))
        print("Second last name : {}".format(secondLastName))
        print("Authentic        : {}".format(authentic))

    def getSegment(self, contours: list, limits: list) -> tuple:
        """ This function takes a list of contours and identifies the full
            segment between the limits specified.

        Args:
            contours (list): contours list.
            limits (list): left, right, upper and lower limit of the
                           target segment.

        Returns:
            tuple: segment coordinates.
        """
        xs = np.array([], dtype=np.uint32)
        ys = np.array([], dtype=np.uint32)
        # Unpacks the limits for the target segment
        xlf, xrl, yul, yll = limits
        # Goes through all the contours
        for (i, c) in enumerate(contours):
            # Unpacks the coordinates of the current segment
            (x1, y1, w, h) = cv2.boundingRect(c)
            # Calculates the x2 and y2 coordinates
            x2 = x1 + w
            y2 = y1 + h
            # If the current segment is inside the limits
            if ((x1 > xlf) and (x2 < xrl) and (y1 > yul) and (y2 < yll) and
                    (w > self.segMinSize) and (h > self.segMinSize)):
                # Stores all the coordinates of the current segment
                xs = np.append(xs, [x1, x2])
                ys = np.append(ys, [y1, y2])
        # Merges the x and y coordinates into one list
        segments = np.array([xs, ys], dtype=np.uint32)
        # Combines the different little segments into a whole segment
        fullSegment = self.putSegmentTogether(contours=segments)

        return fullSegment

    def putSegmentTogether(self, contours: list) -> tuple:
        """ This function takes a list of contours and forms one full contour
            depending on their positions.

        Args:
            contours (list): contours list.

        Returns:
            tuple: full digits contour.
        """
        if (len(contours[0]) > 0 and len(contours[1]) > 0):
            x1 = min(contours[0]) - self.padding
            y1 = min(contours[1]) - self.padding
            x2 = max(contours[0]) + self.padding
            y2 = max(contours[1]) + self.padding
            return (x1, y1, x2, y2)

        return None
    
    def getDocumentType(self, img: np.ndarray, contours: list):
        
        #crop = img[15:65,40:300]
        #cv2.imwrite('test.jpg',crop)
        
        doc = None
        # Gets the coordinates of the id number segment
        docSegment = self.getSegment(contours=contours, limits=[40, 300, 15, 65])
        # If the segment is not None
        if (docSegment):
            # Extracts the id number segment from the input image
            docImg = self.extractROI(img=img, segment=docSegment, zoom=True)
            # Makes the preprocessing of the image
            docdImg = self.im2bin(img=docImg)
            # Applies the OCR algorithm
            doc = self.optRecognizer.im2text(img=docImg)
            # Cleans up the text detected
            doc = re.sub('[^0-9]', '', doc)

        return doc, docSegment         

    def getId(self, img: np.ndarray, contours: list) -> tuple:
        """ This function takes a pre-processed input image of an identification
            card and extracts the identification number information.

        Args:
            img (np.ndarray): input image.
            contours (list): contours list.

        Returns:
            tuple: id number, id image segment
        """
        id = None
        # Gets the coordinates of the id number segment
        idSegment = self.getSegment(contours=contours,
                                    limits=[self.idNumberX1, self.idNumberX2,
                                            self.idNumberY1, self.idNumberY2])
        # If the segment is not None
        if (idSegment):
            # Extracts the id number segment from the input image
            idImg = self.extractROI(img=img, segment=idSegment, zoom=True)
            # Makes the preprocessing of the image
            idImg = self.im2bin(img=idImg)
            # Applies the OCR algorithm
            id = self.optRecognizer.im2text(img=idImg)
            # Cleans up the text detected
            id = re.sub('[^0-9]', '', id)

        return id, idSegment

    def getFullName(self, img: np.ndarray, contours: list) -> tuple:
        """ This function takes a pre-processed input image of an identification
            card and extracts the full name information.

        Args:
            img (np.ndarray): input image.
            contours (list): contours list.

        Returns:
            tuple: full name, name image segment
        """
        fullName = None
        # Gets the coordinates of the full name segment
        fullNameSegment = self.getSegment(contours=contours,
                                          limits=[self.idNameX1, self.idNameX2,
                                                  self.idNameY1, self.idNameY2])
        # If the segment is not None
        if (fullNameSegment):
            # Extracts the full name segment from the input image
            fullNameImg = self.extractROI(img=img, segment=fullNameSegment)
            # Divides the full name into their different parts
            nameImg = self.extractNameROI(img=fullNameImg,
                                          nameCode=0)
            firstLastNameImg = self.extractNameROI(img=fullNameImg,
                                                   nameCode=1)
            secondLastNameImg = self.extractNameROI(img=fullNameImg,
                                                    nameCode=2)
            # Makes the preprocessing of the images
            nameImg = self.im2bin(img=nameImg)
            firstLastNameImg = self.im2bin(img=firstLastNameImg)
            secondLastNameImg = self.im2bin(img=secondLastNameImg)
            # Applies the OCR algorithm
            name = self.optRecognizer.im2text(img=nameImg)
            firstLastName = self.optRecognizer.im2text(img=firstLastNameImg)
            secondLastName = self.optRecognizer.im2text(img=secondLastNameImg)

            # Cleans to leave caracters from A-Z
            name = re.sub('[^a-zA-Z ]+','',name)
            firstLastName = re.sub('[^a-zA-Z ]+','',firstLastName)
            secondLastName = re.sub('[^a-zA-Z ]+','',secondLastName)
            
            # Cleans up the text detected
            name = name.split()
            if (len(name[-1]) == 1):
                name = " ".join(name[:-1])
            else:
                name = " ".join(name)
            firstLastName = firstLastName.split()[0]
            secondLastName = secondLastName.split()[0]
            # Converts all the names letters into uppercase
            name = name.upper()
            firstLastName = firstLastName.upper()
            secondLastName = secondLastName.upper()
            # Concatenates the different name parts
            fullName = "{} {} {}".format(name, firstLastName, secondLastName)

        return fullName, fullNameSegment


    def extractROI(self, img: np.ndarray,
                   segment: tuple,
                   zoom: bool = False,
                   scale: tuple = (1.5, 1.5)) -> np.ndarray:
        """ This function takes an input image and extracts from it the given segment.

        Args:
            img (np.ndarray): input image.
            segment (tuple): segment coordinates.
            zoom (bool, optional): image re-scale. Defaults to False.
            scale (tuple, optional): scaling factor. Defaults to (1.5, 1.5).

        Returns:
            np.ndarray: output image.
        """
        # Extracts the given segment from the input image
        roi = self.imgProcessor.extractROI(img=img,
                                           segment=segment)
        # If the image requires a re-scale
        if (zoom):
            roi = self.imgProcessor.imScale(img=roi,
                                            scale=scale)

        return roi

    def extractNameROI(self, img: np.ndarray, nameCode: int = 0) -> np.ndarray:
        """ This function takes an input image corresponding to a full name
            and extracts the individual name specified.

        Args:
            img (np.ndarray): input image.
            nameCode (int, optional): code defining which name to extract.
                                      Defaults to 0.
                                    - 0     : Name.
                                    - 1     : First last name.
                                    - Other : Second last name.

        Returns:
            np.ndarray: name image
        """
        h, w = img.shape
        nameSize = math.ceil(h/3)
        scale = (3.0, 3.0)
        # Sets the coordinates of the different names
        x1 = 0
        x2 = x1 + w
        y1 = 0
        y2 = y1 + nameSize
        y3 = y2 + nameSize
        y4 = y3 + nameSize
        # Name
        if (nameCode == 0):
            nameSegment = (x1, y1, x2, y2)
            return self.extractROI(img=img,
                                   segment=nameSegment,
                                   zoom=True,
                                   scale=scale)
        # First name
        elif (nameCode == 1):
            firstLastNameSegment = (x1, y2 - 2, x2, y3)
            return self.extractROI(img=img,
                                   segment=firstLastNameSegment,
                                   zoom=True,
                                   scale=scale)
        # Second name
        else:
            secondLastNameSegment = (x1, y3 - 2, x2, y4)
            return self.extractROI(img=img,
                                   segment=secondLastNameSegment,
                                   zoom=True,
                                   scale=scale)

    def im2bin(self, img: np.ndarray) -> np.ndarray:
        """ This function takes an input image and applies to
            it the pre-processing stage for the OCR algorithm.

        Args:
            img (np.ndarray): input image.

        Returns:
            np.ndarray: output image.
        """
        # Reduces the image noise
        blurred = self.imgProcessor.reduceNoise(img=img)
        # Converts the image into binary
        thresh = self.imgProcessor.im2binAdaptive(img=blurred,
                                                  maxValue=128,
                                                  blockSize=15,
                                                  C=16)
        # Performs a morphological opening operation
        opening = self.imgProcessor.morphOpen(img=thresh,
                                              kernel=self.sqKernel3)

        return opening

    def segmentImage(self, img: np.ndarray, k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
        """ This function takes an input image and segments it into its relevant parts,
            that correspond to the text contained in the image.

        Args:
            img (np.ndarray): input image.
            k1 (np.ndarray): black-hat transform kernel.
            k2 (np.ndarray): morphological closing kernel.

        Returns:
            np.ndarray: output image.
        """
        # Performs a black-hat transformation
        blackhat = self.imgProcessor.blackHatTransform(img=img,
                                                       kernel=k1)

        # Converts the image into binary
        thresh = self.imgProcessor.im2bin(img=blackhat)
        # Performs a morphological closing operation
        threshClosed = self.imgProcessor.morphClose(img=thresh,
                                                    kernel=k2)

        return threshClosed