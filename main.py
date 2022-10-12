from CRIdentityCardRecognizer import CRIdentityCardRecognizer
from ImageCapture import ImageCapture
import os
import time
import cv2

capImage = ImageCapture()

# Capture Image  
capImage.captureImage() 

def getFilepaths(directory: str) -> list:
    filePaths = []
    # Goes through all the files in the directories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Joins the two strings in order to form the full file path
            filepath = os.path.join(root, filename)
            # Adds the path to the list
            filePaths.append(filepath)
    return filePaths


dir = "id-validate"
filePaths = getFilepaths(dir)
verbose = True
total_time = 0

recognizer = CRIdentityCardRecognizer()
   

# Goes through all the images in the directory
for fileName in filePaths:
    # Sets the initial time
    start_time = time.time()
    
    recognizer.preprocesImage(filePath=fileName)

    # Applies the algorithm to the current image
    recognizer.verifyIdCardAuthenticity(filePath=fileName, verbose=verbose)
    # Calculates the execution time
    execution_time = time.time() - start_time
    # If verbose is activated shows some process information
    if (verbose):
        print("Execution time   : {:.3f} s".format(execution_time))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print()
    # Adds the current execution time to the general execution time
    total_time += execution_time

print("Total time   : {} s".format(total_time/len(filePaths)))
