import cv2
import numpy as np
import imutils
from collections import defaultdict
from math import atan2, cos, sin, sqrt, pi

class ImageDetection:
    def __init__(self) -> None:
        pass

    def readImage(self, filePath: str) -> np.ndarray:
        "This function takes the path of the image and pass the image info"
        
        return cv2.imread(filePath)
    
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
   
    def grayImage(self, filePath: str) -> None:
        return cv2.cvtColor(src=filePath, code=cv2.COLOR_BGR2GRAY)

    def thresholdImage(self, filePath: str):
        return cv2.threshold(filePath,170, 255, cv2.THRESH_BINARY_INV)[1]

    def apply_adaptive_threshold(self, filePath: str):
        return cv2.adaptiveThreshold(filePath, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 5)

    def countoursImage(self, filePath: str):
        return cv2.findContours(filePath, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    def canny(self, img: np.ndarray):
        canny = cv2.Canny(img, 200,500)
        return cv2.dilate(canny, None, iterations=1)

    def drawContours(self, filePath: str, contours):
        return cv2.drawContours(filePath, contours, -1, (0, 0, 255), 3)
    
    def biggestContour(contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 1000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest, max_area

    def sharpenImagen(self, filePath: str):
        kernel = np.array([[-1, -1, -1],
                            [-1, 9,-1],
                            [-1, -1, -1]])

        return cv2.filter2D(src=filePath, ddepth=-1, kernel=kernel) 

    def ordenar_puntos(self,puntos, angle):
        n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()

        y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
        print('Y \n',y_order)
        
        x1_order = y_order[:2]
        print('X1 \n',x1_order)
        x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])

        x2_order = y_order[2:4]
        print('X2 \n',x2_order)
        x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])
        
        return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

    def ordenar(self, puntos, angle):
        n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
        point_list = np.zeros((4,2))
        x_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[0])
       
        if angle > 0.30:
            point_list[0] = x_order[1]
            point_list[1] = x_order[3]
            point_list[2] = x_order[0]
            point_list[3] = x_order[2] 
            print('Caso 1')
           
        elif angle < -0.30:
            point_list[0] = x_order[0]
            point_list[1] = x_order[2]
            point_list[2] = x_order[1]
            point_list[3] = x_order[3]
            print('Caso 2')
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
            print('Caso 3')

        return point_list

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
      
                if 1 <= aspect_ratio < 2: 
                     
                    #Get orientation angle 
                    angle = self.getOrientation(c) 
                     
                    # Order points acording to the angle 
                    puntosOrdenados = self.ordenar(approx, angle) 
                     
                    pts1 = np.float32(puntosOrdenados) 
                    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]]) 
                     
                    matrix = cv2.getPerspectiveTransform(pts1,pts2) 
 
                    dest = cv2.warpPerspective(filePath, matrix, (width,height)) 

        return dest 

    def imResize(self, img: np.ndarray, width: int, height: int) -> np.ndarray:
        """ This function takes an image and modify their size, without
            loosing the aspect ratio.

        Args:
            img (np.ndarray): input image.
            width (int): new width for the image.

        Returns:
            np.ndarray: output image.
        """
        return imutils.resize(image=img,
                              width=width, height=height)


    def getOrientation(self, pts):
       
        # Construct a buffer used by the pca analysis
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i,0] = pts[i,0,0]
            data_pts[i,1] = pts[i,0,1]
        
        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
        
        # Store the center of the object
        cntr = (int(mean[0,0]), int(mean[0,1]))
        #print('Centro de la imagen: \n', cntr)
        
   
        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        print('angulo: \n',str(-int(np.rad2deg(angle)) - 90))
        
        
        # Label with the rotation angle
        #label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
        #print('angulo: \n', label)
        #cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        #cv2.imwrite('pruebas/angulo.jpg',img)

        return angle
        