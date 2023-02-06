import base64
import pickle
from pyzbar.pyzbar import decode
import cv2  # pip install opencv-python
import numpy as np
from typing import Any, Dict, List, Tuple
from interface import RecognizerInterface
from platform import python_version
import pytesseract  # ocr p/ characteres recognize https://github.com/UB-Mannheim/tesseract/wiki
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

print(f"This was made in python 3.10.5")
print(f"Your python version is: {python_version()}")
STUDENT_ID = 0


class ImageBody(BaseModel):
    """
    A imagem recebida está em base64
    Sendo,portanto, enviada no body da requisição
    """
    image: str


pytesseract.pytesseract.tesseract_cmd = r'./Tesseract-OCR/tesseract'  # executable


class TheRecognizer(RecognizerInterface):
    # pip install "uvicorn[standard]" e pip install fastapi
    APP = FastAPI()
    # posicoes
    options: list[hex] = [
        0x10, 0x11, 0x12,
        0x13, 0x14
    ]

    def __init__(self, image_path: str, question_number: int) -> None:
        """
        :param image_path: caminho da imagem
        :param question_number: numero de questões por linha, exemplo A, B => 2
        """
        self.__image_path: str = image_path
        self.__question_number: int = question_number
        self.__STUDENT_ID = self.capture_the_student_registration(self.__image_path)

    def view_test(self, image) -> None:
        """
        Apenas visualiza o resultado intermediário
        :param image:
        :return:
        """
        img = cv2.resize(image, (500, 600))
        cv2.imshow('view_image.jpg', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_retangle(self, img) -> Tuple:
        """
        Obtem retangulo
        :param img:
        :return:
        """
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 12)
        cv2.medianBlur(image_thresh, 5)
        kernel = np.ones((2, 2), np.uint8)
        image_dilate = cv2.dilate(image_thresh, kernel)
        contours, _ = cv2.findContours(image_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contourn = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contourn)
        bbox = [x, y, w, h]
        cv2.rectangle(image_dilate, (x, y), (x + w, y + h), (255, 0, 0), 4)
        return img, bbox

    def make_pickle_file(self, image) -> List:
        """
        Executar primeiro, apenas uma vez!
        :param image:
        :return:
        """
        spaces: list = list()

        for marks in range(240):
            space = cv2.selectROI('mark the spaces', image, False)
            cv2.destroyWindow('mark the spaces')
            spaces.append(space)

            for x, y, width, height in spaces:
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 3)
        return spaces

    def save_to_pickle_file(self, image) -> None:
        """
        Salva pickle
        :param image:
        :return:
        """
        spaces = self.make_pickle_file(image)
        with open('positions.pkl', 'wb') as file:
            pickle.dump(spaces, file)

    def finder(self, img) -> Dict:
        """
        encontra as bolhas
        :param img:
        :return:
        """
        final_answers = list()
        path = img
        img = cv2.imread(img)

        gabarito, bbox = self.get_retangle(img)

        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, image_thresh = cv2.threshold(image_gray, 70, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        with open('positions.pkl', 'rb') as file:
            area = pickle.load(file)
        variable = 200
        answers = []
        for id, bubbles in enumerate(area):
            x = int(bubbles[0])
            y = int(bubbles[1])
            _width = int(bubbles[2])
            _height = int(bubbles[3])
            cv2.rectangle(img, (x, y), (x + _width, y + _height), (0, 0, 0), 2)
            cv2.rectangle(image_thresh, (x, y), (x + _width, y + _height), (255, 255, 255), 1)
            camp = image_thresh[y:y + _height, x:x + _width]
            height, width = camp.shape[:2]
            size = height * width
            black_target = cv2.countNonZero(camp)
            if black_target > 500:
                variable = 500
            percent = round((black_target / size) * 100, 2)
            #self.view_test(img) # DEBUG
            if percent >= 20:
                cv2.rectangle(img, (x, y), (x + _width, y + _height), (0, 0, 255), 2)
                answers.append(area[id])
        i = 0
        for resp in answers:
            for camp in area:
                if resp == camp:
                    if chr(int(f'{self.options[i]}', 10) + 49) is not None:
                        final_answers.append(chr(int(f'{self.options[i]}', 10) + 49))
                i += 1
                if i == 4:
                    i = 0
        keys = list(range(1, len(final_answers) + 1))
        json_answers = dict()
        json_answers['student_registration'] = STUDENT_ID
        for index, content in enumerate(final_answers):
            json_answers[keys[index]] = content
        print(json_answers)
        return json_answers

    def start_reconnaissance(self) -> Dict:
        return self.__find_appointment()

    @staticmethod
    def capture_the_student_registration(path_from_image: str) -> None:
        global STUDENT_ID
        img = cv2.imread(path_from_image)
        for code in decode(img):
            STUDENT_ID = code.data.decode()
            break

    @staticmethod
    @APP.post('/recognizer')
    @APP.post('/therecognizer')
    def create_a_endpoint_to_server_application(image: ImageBody) -> JSONResponse:
        """
        Renderiza um endpoint para a aplicação
        documentação em: /docs
        to run:  uvicorn Recognizer:TheRecognizer.APP --reload
        :return: JSONResponse
        """
        image_decode_base64 = base64.b64decode(image.image)
        with open(f'images_to_scan/recognizer_image2.jpeg', 'wb') as img:
            img.write(image_decode_base64)
        recognizer = TheRecognizer("images_to_scan/recognizer_image2.jpeg",
                                   10)  # 525x700  images teste4 e gab (preferível png)
        json_compatible = jsonable_encoder(recognizer.finder("images_to_scan/recognizer_image2.jpeg"))
        return JSONResponse(content=json_compatible)

    @APP.get('/recognizer')
    def create_a_endpoint_to_server_application(self) -> JSONResponse:
        """
        Renderiza um endpoint para a aplicação
        documentação em: /docs
        to run:  uvicorn Recognizer:TheRecognizer.APP --reload
        :return: JSONResponse
        """
        json_compatible = jsonable_encoder({"message": "Por favor, use o método post com a imagem no body. img=image"})
        return JSONResponse(content=json_compatible)

    @APP.exception_handler(StarletteHTTPException)
    def page_not_found(request: Request, exc: Any) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={f"Recognizer© says: ERROR [{str(exc.detail)}]": "Opa! Você tentou acessar um endereço inválido "
                                                                     "ou houve um problema interno "
                                                                     ". Por favor,tente: /recognizer"}
        )

    def __repr__(self) -> str:
        return "Recognizer©"


if __name__ == '__main__':
    """
    Para executar, ele usa o servidor uvicorn, que é uma das implementações
    mais rápidas do python, no que diz respeito a requisições HTTP.     
    Logo, basta fazer:                                   
    uvicorn Recognizer:TheRecognizer.APP --reload 
    e acessar a url: http://endereço:8000/recognizer               
    A documentação desse endpoint está        em /     docs     
    """
    try:
        recog = TheRecognizer('images_to_scan/ifmg_gabarito.png', 240)
        recog.finder('images_to_scan/ifmg_gabarito2.jpg')
    except Exception as e:
        print("[ERROR] Verifique se o nome do arquivo esta correto")
