# Identification Card Authenticator

1. Install Tesseract on the computer, as shown [here](https://stackoverflow.com/questions/46140485/tesseract-installation-in-windows).
2. Install the required python libraries using the following command.
    
    `pip install opencv-python imutils numpy pandas tesseract PIL Levenshtein fuzzywuzzy`
    
3. Download the text file containing all the database information from [here](https://www.tse.go.cr/descarga_padron.htm) and, put it inside a folder named _datasets_.
4. Put the images inside a folder named _id-validate_.
5. Run the main file using the following command.

    `python main.py`
