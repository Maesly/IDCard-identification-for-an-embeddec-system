import cv2
import numpy as np
import imutils


class ImageCapture:

    def __init__(self, camera_id = '/dev/video0') -> None:
        self.camera_id = camera_id
        self.heightImg = 550
        self.widthImg = 88
        self.count = 0
        self.webcam = True
        self.cap = cv2.VideoCapture(self.camera_id,cv2.CAP_V4L2)
    
    def isOpen(self) -> bool:
        open = True
        if not self.cap.isOpened():
            open = False
            raise IOError("Cannot open webcam")
        return open

    def captureImage(self):
        while self.webcam:
            ret, frame = self.cap.read()
            
            frame = cv2.resize(frame, (640, 480))
            
            cv2.imshow('ID-Card recognition', frame)
            c = cv2.waitKey(1)
            if c == 115:
                cv2.imwrite('id-validate/image.jpg',frame)
                #cv2.imwrite('Pruebas/image.jpg',frame)
                print('Image successfully captured')
                break


        self.cap.release()
        cv2.destroyAllWindows()

        