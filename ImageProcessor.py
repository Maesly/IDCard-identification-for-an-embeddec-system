from turtle import width
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
from collections import defaultdict
from math import atan2, cos, sin, sqrt, pi

class ImageProcessor:

    def __init__(self, ratio = 1.54, max_w = 80, min_w = 110, max_h = 25, min_h = 52):
        self.ratio = ratio
        self.max_w = max_w
        self.max_h = max_h
        self.min_w = min_w
        self.min_h = min_h
        pass

    def readImage(self, filePath: str) -> np.ndarray:
        """ This function takes the path of an image file and loads their
            information into a matrix.

        Args:
            filePath (str): image file path.

        Returns:
            np.ndarray: output image.
        """
        return cv2.imread(filePath)

    def imResize(self, img: np.ndarray, width: int) -> np.ndarray:
        """ This function takes an image and modify their size, without
            loosing the aspect ratio.

        Args:
            img (np.ndarray): input image.
            width (int): new width for the image.

        Returns:
            np.ndarray: output image.
        """
        return imutils.resize(image=img,
                              width=width)

    def imScale(self, img: np.ndarray, scale: tuple) -> np.ndarray:
        """ This function takes an image and scales its size.

        Args:
            img (np.ndarray): input image.
            scale (tuple): scale factor along each axis (x, y).

        Returns:
            np.ndarray: output image.
        """
        return cv2.resize(src=img,
                          dsize=None,
                          fx=scale[0],
                          fy=scale[1],
                          interpolation=cv2.INTER_CUBIC)

    def im2grayscale(self, img: np.ndarray) -> np.ndarray:
        """ This function takes and RGB image and converts it into
            a grayscale image.

        Args:
            img (np.ndarray): input image.

        Returns:
            np.ndarray: output image.
        """
        return cv2.cvtColor(src=img,
                            code=cv2.COLOR_BGR2GRAY)

    def im2bin(self, img: np.ndarray, thresh: int = 0, maxval: int = 255) -> np.ndarray:
        """ This function takes an image and converts it into a binary image.

        Args:
            img (np.ndarray): input image.
            thresh (int, optional): threshold value. Defaults to 0.
            maxval (int, optional): maximum value to use. Defaults to 255.

        Returns:
            np.ndarray: output image.
        """
        return cv2.threshold(src=img,
                             thresh=thresh,
                             maxval=maxval,
                             type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def im2binAdaptive(self, img: np.ndarray, maxValue: int, blockSize: int, C: int) -> np.ndarray:
        """ This function takes an image and converts it into a binary image
            using an adaptive thresholding.

        Args:
            img (np.ndarray): input image.
            maxValue (int): threshold value.
            blockSize (int): size of the neighborhood area.
            C (int): constant to subtract from the weighted mean calculated.

        Returns:
            np.ndarray: output image.
        """
        return cv2.adaptiveThreshold(src=img,
                                     maxValue=maxValue,
                                     adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                     thresholdType=cv2.THRESH_BINARY,
                                     blockSize=blockSize,
                                     C=C)

    def reduceNoise(self, img: np.ndarray, ksize: tuple = (3, 3), sigmaX: int = 0) -> np.ndarray:
        """ This function takes an image and reduces the noise present in it by using a
            Gaussian filter.

        Args:
            img (np.ndarray): input image.
            ksize (tuple, optional): size of the neighborhood area. Defaults to (3, 3).
            sigmaX (int, optional): standard deviation of the Gaussian kernel. Defaults to 0.

        Returns:
            np.ndarray: output image.
        """
        return cv2.GaussianBlur(src=img,
                                ksize=ksize,
                                sigmaX=sigmaX)

    def blackHatTransform(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """ This funcion takes an image and applies to it and black-hat tranform to it.

        Args:
            img (np.ndarray): input image.
            kernel (np.ndarray): structuring element.

        Returns:
            np.ndarray: output image.
        """
        return cv2.morphologyEx(src=img,
                                op=cv2.MORPH_BLACKHAT,
                                kernel=kernel)

    def detectEdges(self, img: np.ndarray, dx: int, dy: int, ksize: int) -> np.ndarray:
        """ This function takes an image and applies to it an edge detection algorithm
            using a Sobel operator.

        Args:
            img (np.ndarray): input image.
            dx (int): dx order of the derivative x.
            dy (int): dy order of the derivative y.
            ksize (int): size of the extended Sobel kernel; it must be 1, 3, 5, or 7.

        Returns:
            np.ndarray: output image.
        """
        gradX = cv2.Sobel(src=img,
                          ddepth=cv2.CV_32F,
                          dx=dx,
                          dy=dy,
                          ksize=ksize)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        return gradX

    def morphClose(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """ This function takes an image and applies to it a morphological
            closing operation.

        Args:
            img (np.ndarray): input image.
            kernel (np.ndarray): structuring element.

        Returns:
            np.ndarray: output image.
        """
        return cv2.morphologyEx(src=img,
                                op=cv2.MORPH_CLOSE,
                                kernel=kernel)

    def morphOpen(self, img: np.ndarray, kernel: np.ndarray, iters: int = 1) -> np.ndarray:
        """ This function takes an image and applies to it a morphological
            opening operation.

        Args:
            img (np.ndarray): input image.
            kernel (np.ndarray): structuring element.
            iters (int): number of times erosion and dilation are applied. Defaults to 1.

        Returns:
            np.ndarray: output image.
        """
        return cv2.morphologyEx(src=img,
                                op=cv2.MORPH_OPEN,
                                kernel=kernel,
                                iterations=iters)

    def findContours(self, img: np.ndarray) -> np.ndarray:
        """ This function takes an image and, identifies and returns the contours
            of image segments.

        Args:
            img (np.ndarray): input image.

        Returns:
            np.ndarray: detected contours.
        """
        cnts = cv2.findContours(image=img,
                                mode=cv2.RETR_EXTERNAL,
                                method=cv2.CHAIN_APPROX_SIMPLE)

        return sorted(imutils.grab_contours(cnts),
                      key=cv2.contourArea,
                      reverse=True)

    def extractROI(self, img: np.ndarray, segment: tuple) -> np.ndarray:
        """ This function takes an image and extract the specified region of interest.

        Args:
            img (np.ndarray): input image.

        Returns:
            np.ndarray: output image.
        """
        x1, y1, x2, y2 = segment

        return img[y1:y2, x1:x2]

    def drawSegments(self, img: np.ndarray,
                     segments: list,
                     color: tuple = (255, 0, 0),
                     thickness: int = 1) -> np.ndarray:
        """ This function takes an image and draws all the segments contained
            in the given list.

        Args:
            img (np.ndarray): input image.
            segments (list): segments to draw.
            color (tuple, optional): segment color. Defaults to (255, 0, 0).
            thickness (int, optional): segment thickness. Defaults to 1.

        Returns:
            np.ndarray: output image.
        """
        # Goes through the all the segments
        for i in range(len(segments)):
            img = self.drawSegment(img=img, segment=segments[i])

        return img

    def drawSegment(self, img: np.ndarray,
                    segment: list,
                    color: tuple = (255, 0, 0),
                    thickness: int = 1) -> np.ndarray:
        """ This function takes an image and draws in it the given segment.

        Args:
            img (np.ndarray): input image.
            segment (list): segment to draw.
            color (tuple, optional): segment color. Defaults to (255, 0, 0).
            thickness (int, optional): segment thickness. Defaults to 1.

        Returns:
            np.ndarray: output image.
        """
        x1, y1, x2, y2 = segment
        return cv2.rectangle(img=img,
                             pt1=(x1, y1),
                             pt2=(x2, y2),
                             color=color,
                             thickness=thickness)

    def automatic_brightness_and_contrast(self, img: np.ndarray)-> np.ndarray:
        """ This function takes an image and improve its brightness and contrast.

        Args:
            img (np.ndarray): input image.

        Returns:
            np.ndarray: output image.
        """
        clip_hist_percent=1
        
        # Calculate grayscale histogram
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        hist_size = len(hist)
        
        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))
        
        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0
        
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
        
        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
        
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return auto_result

    def apply_adaptive_threshold(self, filePath: str):

        """ This function takes an image and applies threshold filter

        Args:
            filePath (str): image file path
        Returns:
            np.ndarray: output image.
        """
        return cv2.adaptiveThreshold(filePath, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 25)

    def canny(self, filePath: np.ndarray):
        """ This function takes an image and detect all the edges

        Args:
            filePath (str): image file path
        Returns:
            np.ndarray: output image.
        """
        canny = cv2.Canny(filePath, 200,500)
        return cv2.dilate(canny, None, iterations=1)

    def countoursImage(self, filePath: str):
        """ This function takes an image and found all the contours 

        Args:
            filePath (str): image file path
        Returns:
            np.ndarray: output of the contours.
        """
        return cv2.findContours(filePath, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    def drawContours(self, filePath: str, contours):
        """ This function takes an image and draw the contours that appears on the image

        Args:
            filePath (str): image file path
            contours (ndarray): list of all the contours found on the image
        Returns:
            np.ndarray: output image.
        """
        return cv2.drawContours(filePath, contours, -1, (0, 0, 255), 3)
    
    def sharpenImagen(self, filePath: str):
        """ This function takes an image and sharpen the image

        Args:
            filePath (str): image file path
        Returns:
            np.ndarray: output image.
        """
        kernel = np.array([[-1, -1, -1],
                            [-1, 9,-1],
                            [-1, -1, -1]])
        return cv2.filter2D(src=filePath, ddepth=-1, kernel=kernel) 

    def orderPoints(self, puntos, angle):
        """ This function takes a list of points and reorder the points 
            on the axis x or y according angle of the object

        Args:
            puntos(ndarray): list of all the points that were detected on an image
            angle(float): angle of the position of the object detected on the image
        Returns:
            point_list(ndarray): organized list of the points 
        """
        n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
        point_list = np.zeros((4,2))
        x_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[0])
       
        # Case 1: if the ID card has an inclination to the right

        if angle > 0.30:
            point_list[0] = x_order[1]
            point_list[1] = x_order[3]
            point_list[2] = x_order[0]
            point_list[3] = x_order[2] 
            
        # Case 2: if the ID card has an inclination to the left   
           
        elif angle < -0.30:
            point_list[0] = x_order[0]
            point_list[1] = x_order[2]
            point_list[2] = x_order[1]
            point_list[3] = x_order[3]

        # Case 3: if the ID card has no inclination   

        else:
            y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])

            x1_order = y_order[:2]
            x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])

            x2_order = y_order[2:4]
            x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])
 
            point_list[0] = x1_order[0]
            point_list[1] = x1_order[1]
            point_list[2] = x2_order[0]
            point_list[3] = x2_order[1]

        return point_list


    def biggestContour(self, contour: np.ndarray):
        """ This function takes all the contours that were 
            detected on an image and return the biggest contours that there is

        Args:
            contours (ndarray):array with all the contours detected on an image

        Returns:
            biggest (np.array): array with the biggest contour
            max_area (int): area of the biggest contour     
        """
        biggest = np.array([])
        max_area = 0
        for i in contour:
            area = cv2.contourArea(i)
            if area > 5000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest, max_area

    def getOrientation(self, pts: np.ndarray):
        """ This function takes a contour and gets the angle of the ID card

        Args:
            pts (np.ndarray): contour points
        Returns:
            float: angle of the ID card
        """       
        # Construct a buffer used by the pca analysis
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i,0] = pts[i,0,0]
            data_pts[i,1] = pts[i,0,1]
        
        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

        # Orientation in radians
        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) 
        return angle
 
    def cutImage(self, filePath: np.ndarray, contours: np.ndarray):
        """ This function takes an image an cut out the image at the edges of the ID card

        Args:
            filePath (np.ndarray): image file path
            contours: all the contours found on the image

        Returns:
            np.ndarray: output image.
        """
        dest = np.copy(filePath) 
     
        # Width of ID Card is 88mm  
        width = 880 
        # Height of ID Card is 55mm  
        height = 550
        cont = 0
        for c in contours: 
            area = cv2.contourArea(c) 
 
            # Takes all the points of the figures 
            x,y,w,h = cv2.boundingRect(c) 
            epsilon = 0.09*cv2.arcLength(c,True) 
             
            # Approx counts how many points the detected figure has 
            approx = cv2.approxPolyDP(c,epsilon,True) 
             
            if len(approx) == 4 and area > 10000:      
                 
                # aspect ratio is the approximate area of the IDCard. Ideal: 880/550 = 1.6                    
                aspect_ratio = float(w)/h                 
      
                if 1 <= aspect_ratio < 2 and cont == 0: 
                     
                    #Get orientation angle 
                    angle = self.getOrientation(c) 
                     
                    # Order points acording to the angle 
                    puntosOrdenados = self.orderPoints(approx, angle) 
                     
                    pts1 = np.float32(puntosOrdenados) 
                    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]]) 
                     
                    matrix = cv2.getPerspectiveTransform(pts1,pts2) 
 
                    dest = cv2.warpPerspective(filePath, matrix, (width,height)) 
                    cont +=1
        return dest 

    def histogram(self, filePath: np.ndarray): 
        return cv2.equalizeHist(filePath)
