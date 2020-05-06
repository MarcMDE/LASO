import speech_recognition as sr
import sys

class VoiceRecognition:
    def __init__(self, restart):
        self.restart=restart


    def recon_Voice(self):
        r = sr.Recognizer()
        coord=[]
        correcto1=1
        correcto2=1
        accio = "none"

        with sr.Microphone() as source:
            print("Speak")
            try:
                audio = r.listen(source)
                text = r.recognize_google(audio, language='es-ES')
                print(text)

                if 'empezar' == text or 'comenzar' == text:
                    print("Game is runing")
                    accio = "a1"
                elif 'terminar' == text or 'parar' == text:
                        print("Game finished")
                        accio = "a2"
                elif 'coordenada' == text or 'coordenadas' == text:
                    accio = "a3"

                    while correcto1:
                        cX=input("Escribe la coordenada X: ")
                        if int(cX) <= 100 and int(cX) >= 0:
                            cY=input("Escribe la coordenada Y: ")
                            if int(cY) <= 100 and int(cY) >= 0:
                                correcto1=0
                                coord = [int(cX), int(cY)]
                        """with sr.Microphone() as source:
                            print("Di la coordenada X: ")

                            try:
                                audioX = r.listen(source)
                                textX = r.recognize_google(audioX, language='es-ES')

                                if textX <= 80 and textX >= 0: #Canviar numeros
                                    correcto1=0

                                    while correcto2:
                                        with sr.Microphone() as source:
                                            print("Di la coordenada Y: ")

                                            try:
                                                audioY = r.listen(source)
                                                textY = r.recognize_google(audioY, language='es-ES')

                                                if textY <= 80 and textY >= 0: #canviar numeros
                                                    correcto2=0
                                                    coord=[int(textX), int(textY)]

                                            except sr.UnknownValueError:
                                                print('error')
                                            except sr.RequestError as e:
                                                print('failed'.format(e))

                            except sr.UnknownValueError:
                                print('error')
                            except sr.RequestError as e:
                                print('failed'.format(e))"""

                elif 'reiniciar' == text or 'reinicia' == text:
                    accio = "a4"

                else:
                     print("No hay comando asociado a esta palabra")

            except sr.UnknownValueError:
                print('Error, no se entendio la palabra')
                return accio, coord
            except sr.RequestError as e:
                print('failed'.format(e))
                return accio, coord
        return accio,coord



