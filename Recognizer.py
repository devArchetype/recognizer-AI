import base64
import cv2  # pip install opencv-python
import imutils
from typing import Any, Dict
import numpy as np
from imutils import contours
from platform import python_version
from imutils.perspective import four_point_transform
import pytesseract  # an ocr for characteres recognize https://github.com/UB-Mannheim/tesseract/wiki
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
print(f"This was made in python 3.10.5")
print(f"Your python version is: {python_version()}")
print("REQUIREMENTS:")
print("Image with a big dark background")


class ImageBody(BaseModel):
    """
    A imagem recebida está em base64
    Sendo,portanto, enviada no body da requisição
    """
    image: str


pytesseract.pytesseract.tesseract_cmd = r'./Tesseract-OCR/tesseract'  # executable


class TheRecognizer:
    """
        Recognizer©
        É um projeto sobre visão computacional e processamento de imagens com o intuito de resolver
        problemas de forma prática e acessível.

        Este projeto deve resolver as demandas escolares que,a nós, foram solicitadas.

        Conceitos envolvidos:
            OMR (Optical Mark Recognition) reconhecedor de marca
            optica que é uma forma de reconhecer marcas e assinalações
            em documentos, por humanos.

            Lógica de implementação:
            - Aplicar transformação de perspectiva
            - extrair a primeira fileira de respostas
            - determinar quais foram as opções marcadas
            - repetir o algoritmo para cada fileira

         A começar, ao Processar a imagem selecionada, é importante convertê-la para escala cinza,
         para um melhor funcionamento da lib

        Além disso, é super importante obter, antes de tudo, a borda do documento para aplicar a
        transformação de perspectiva

        Feito isso, para classificar o gabarito, é necessário aplicar a binarização ou delimitação/segmentação
        Com a imagem binarizada(totalmente preta com contornos brancos), é necessário encontrar os contornos novamente.

        Logo, é necessário classificar as perguntas de cima para baixo, para que as questoes fiquem
        na ordem que aparece o gabarito.

        Após ordernar de cima para baixo o gabarito, é importante garantir que as respostas estarão da esquerda para
        a direita

        Com as bolhas do gabarito encontradas, é necessário saber qual está colorida,para isso, basta ver qual está
        com pixel próximo a zero, ou seja, branco.

        Para finalizar, para encontrar as respostas marcadas, deve-se identificar os pixels diferentes de 0, ou seja,
        que possuem alguma marcação.Aqueles que tiverem a maior porcentagem, devem estar marcados e serão considerados,
        daí, a importância de estar bem preenchido o cartão.
    """

    APP = FastAPI()  # pip install "uvicorn[standard]" e pip install fastapi

    def __init__(self, image_path: str, question_number: int) -> None:
        """
        :param image_path: caminho da imagem
        :param question_number: numero de questões por linha, exemplo A, B => 2
        """
        self.__image_path: str = image_path
        self.__question_number: int = question_number
        self.__STUDENT_ID = self.capture_the_student_registration(self.__image_path)

    def view_test(self, image) -> None:
        """ Apenas visualiza o resultado intermediário """
        cv2.imshow('view_image.jpg', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __pre_process_image(self) -> tuple:
        input_image = cv2.imread(self.__image_path)
        gray_color = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        blur_image = cv2.GaussianBlur(gray_color, (5, 5), 0)  # reduz o ruído de frequência
        edged = cv2.Canny(blur_image, 75, 200)  # encontra as bordas do gabarito
        return edged, input_image, gray_color, blur_image

    def __find_contours(self) -> Any:
        """Encontra o contorno externo,isto é, do documento, em si"""
        contour = cv2.findContours(self.__pre_process_image()[0].copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        c = None
        if len(contour) > 0:  # se existirem, é retornado o par ordenado com os pontos do contorno
            # analisa o contorno do gabarito, colocando os de maior area na frente (quadro geral)
            contour = sorted(contour, reverse=True, key=cv2.contourArea)
            for points in contour:
                perimeter = cv2.arcLength(points, True)
                approximation = cv2.approxPolyDP(points, 0.02 * perimeter, True)
                if len(approximation) == 4:  # 4 vértices, já que o gabarito, em si, é um retângulo
                    c = approximation
                    return c
        return c

    def __apply_perspective(self) -> Any:
        original_image = self.__pre_process_image()[1]
        gray_image = self.__pre_process_image()[2]
        # aplicando transformação de perspectiva (para ler o documento de cima para baixo) [90º]
        # four1 = four_point_transform(original_image, self.__find_contours().reshape(4, 2))
        four2 = four_point_transform(gray_image, self.__find_contours().reshape(4, 2))
        binarization_of_segmentation = cv2.threshold(four2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # self.view_test(binarization_of_segmentation)
        return binarization_of_segmentation

    def __find_contours_in_thresholded_image(self) -> tuple:
        questions = list()
        contours_ = cv2.findContours(self.__apply_perspective().copy(),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        contours_ = imutils.grab_contours(contours_)
        for points in contours_:
            (x, y, w, h) = cv2.boundingRect(points)  # encontra os contornos
            ar = w / float(h)
            # encontra o contorno como uma regiao alta de proporcao = 1
            if w >= 20 and h >= 20 and 0.9 <= ar <= 1.3:  # ve qual é o circulo da opcao
                questions.append(points)
        return questions, contours_

    def __classification_of_questions_to_top_to_bottom(self) -> tuple:
        questions = self.__find_contours_in_thresholded_image()[0]
        contours_ = self.__find_contours_in_thresholded_image()[1]
        questions = contours.sort_contours(questions,
                                           method="top-to-bottom")  # basicamente, aqui, ja se identifica os pixels em que se encontram as bolhas
        return questions[0], contours_

    def __find_appointment(self) -> Dict:
        """Inicia processo de reconhecimento"""
        answers = dict()
        answers['student_registration'] = self.__STUDENT_ID
        questions, contours_ = self.__classification_of_questions_to_top_to_bottom()
        count: int = 1  # questao começa em 1
        for question_index, question in enumerate(
                np.arange(0, len(questions), self.__question_number)):  # o numero de questoes
            contours_ = contours.sort_contours(questions[question:question + self.__question_number])[
                0]  # classifica o gabarito da esquerda para a direita
            bubble = None
            for index, contor in enumerate(contours_):
                mask = np.zeros(self.__apply_perspective().shape, dtype="uint8")
                cv2.drawContours(mask, [contor], -1, 255, -1)
                #self.view_test(cv2.drawContours(mask, [contor], -1, 0, -1))
                # aplica mascara e conta o numero de nao zeros
                mask = cv2.bitwise_and(self.__apply_perspective(), self.__apply_perspective(), mask=mask)
                total_non_zeros = cv2.countNonZero(mask)
                # se o total for maior que o total de pixels diferentes de zero,entao,é a resposta
                if bubble is None or total_non_zeros > bubble[
                    0]:  # opcoes marcadas possuem mais numeros diferentes de zero
                    bubble = (total_non_zeros, index)
            answers[count] = chr(bubble[1] + 65)
            count += 1
        return answers

    def start_reconnaissance(self) -> Dict:
        return self.__find_appointment()

    @staticmethod
    def capture_the_student_registration(path_from_image: str) -> int:
        with Image.open(path_from_image) as img:
            return int(pytesseract.image_to_string(img))

    @staticmethod
    @APP.post('/recognizer')
    def create_a_endpoint_to_server_application(image: ImageBody) -> JSONResponse:
        """
        Renderiza um endpoint para a aplicação
        documentação em: /docs
        to run:  uvicorn Recognizer:TheRecognizer.APP --reload
        :return: JSONResponse
        """
        image_decode_base64 = base64.b64decode(image.image)
        with open(f'images_to_scan/recognizer_image.jpeg', 'wb') as img:
            img.write(image_decode_base64)
        recognizer = TheRecognizer("images_to_scan/recognizer_image.jpeg", 5)  # 525x700  images teste4 e gab (preferível png)
        json_compatible = jsonable_encoder(recognizer.start_reconnaissance())
        return JSONResponse(content=json_compatible)

    @APP.get('/recognizer')
    def create_a_endpoint_to_server_application(self) -> JSONResponse:
        """
        Renderiza um endpoint para a aplicação
        documentação em: /docs
        to run:  uvicorn Recognizer:TheRecognizer.APP --reload
        :return: JSONResponse
        """
        json_compatible = jsonable_encoder({"message":"Por favor, use o método post com a imagem no body. img=image"})
        return JSONResponse(content=json_compatible)

    @APP.exception_handler(StarletteHTTPException)
    def page_not_found(request: Request, exc: Any) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={f"Recognizer© says: ERROR [{str(exc.detail)}]": "Opa! Você tentou acessar um endereço inválido ou houve um problema interno"
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
    A documentação desse endpoint está em /docs
    """
