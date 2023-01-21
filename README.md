# Recognizer 

Recognizer©<br>
        👁 <p>It is a project about computer vision and image processing in order to solve<br>
        problems in a practical and accessible way.<br>
        This project must solve the school demands that were asked of us.<br><br>
        Concepts involved:<br>
        &#09;    OMR (Optical Mark Recognition) mark recognizer<br>
        &#09;    optica which is a way to recognize marks and markings<br>
         &#09;   in documents, by humans.<br>
         &#09;   Implementation logic:<br>
        &#09;   - Apply perspective transformation<br>
       &#09;     - extract the first row of answers<br>
        &#09;    - determine which options were checked<br>
        &#09;    - repeat the algorithm for each row
        <br><br>
         To begin with, when Processing the selected image, it is important to convert it to grayscale,<br>
         for a better functioning of the lib<br>
        Also, it's super important to get, first of all, the edge of the document to apply the<br>
        perspective transformation<br>
        Once this is done, to classify the template, it is necessary to apply binarization or delimitation/segmentation<br>
        With the image binarized (completely black with white outlines), it is necessary to find the outlines again.<br>
        Therefore, it is necessary to sort the questions from top to bottom, so that the questions are<br>
        in the order in which the template appears.<br>
        After ordering the answer sheet from top to bottom, it is important to ensure that the answers are from left to<br>
        on the right<br>
        With the template bubbles found, it is necessary to know which one is colored, for that, just see which one is<br>
        with pixel close to zero, that is, white.<br>
        Finally, to find the marked answers, you must identify the pixels different from 0, that is,<br>
        that have some marking. Those that have the highest percentage, must be marked and will be considered,<br>
        hence, the importance of filling out the card correctly.</p><br>

<br>

![image](https://user-images.githubusercontent.com/88283829/212799817-bc727cc6-e35a-4ba7-a974-0f44461d27f1.png)

<br>
<hr>

### ❗ Note: Some processing is performed by: 

        https://github.com/tesseract-ocr/tesseract 

Because, Tesseract 4 adds a new neural net (LSTM) based OCR engine which is focused on line recognition, but also still supports the legacy Tesseract OCR engine of Tesseract 3 which works by recognizing character patterns. Compatibility with Tesseract 3 is enabled by using the Legacy OCR Engine mode (--oem 0). It also needs traineddata files which support the legacy engine, for example those from the tessdata repository.

<hr>  

### 🐍 Python version:

        3.10.5

### 🖥️ How to use:

       > pip install -r requirements
       
       > uvicorn Recognizer:TheRecognizer.APP --reload 


### Response:

![image](https://user-images.githubusercontent.com/88283829/213879047-16575227-87a9-42a2-9542-066720321e49.png)

       
