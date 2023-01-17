# Recognizer 

Recognizer©
        It is a project about computer vision and image processing in order to solve
        problems in a practical and accessible way.
        This project must solve the school demands that were asked of us.
        Concepts involved:
            OMR (Optical Mark Recognition) mark recognizer
            optica which is a way to recognize marks and markings
            in documents, by humans.
            Implementation logic:
            - Apply perspective transformation
            - extract the first row of answers
            - determine which options were checked
            - repeat the algorithm for each row
         To begin with, when Processing the selected image, it is important to convert it to grayscale,
         for a better functioning of the lib
        Also, it's super important to get, first of all, the edge of the document to apply the
        perspective transformation
        Once this is done, to classify the template, it is necessary to apply binarization or delimitation/segmentation
        With the image binarized (completely black with white outlines), it is necessary to find the outlines again.
        Therefore, it is necessary to sort the questions from top to bottom, so that the questions are
        in the order in which the template appears.
        After ordering the answer sheet from top to bottom, it is important to ensure that the answers are from left to
        on the right
        With the template bubbles found, it is necessary to know which one is colored, for that, just see which one is
        with pixel close to zero, that is, white.
        Finally, to find the marked answers, you must identify the pixels different from 0, that is,
        that have some marking. Those that have the highest percentage, must be marked and will be considered,
        hence, the importance of filling out the card correctly.


<hr>

Note: Some processing is performed by: <br>
https://github.com/tesseract-ocr/tesseract <br>

Because, Tesseract 4 adds a new neural net (LSTM) based OCR engine which is focused on line recognition, but also still supports the legacy Tesseract OCR engine of Tesseract 3 which works by recognizing character patterns. Compatibility with Tesseract 3 is enabled by using the Legacy OCR Engine mode (--oem 0). It also needs traineddata files which support the legacy engine, for example those from the tessdata repository.

<hr>
