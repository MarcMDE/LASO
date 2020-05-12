import speech_recognition as sr
import sys

class VoiceRecognition:
    def __init__(self, restart):
        self.restart=restart
        self.voice_solving = False
        self.dir = (0,0)


    def recon_Voice(self):
        r = sr.Recognizer()
        coord=None
        correcto1=1
        correcto2=1
        accio = "none"

        with sr.Microphone() as source:
            print("Speak")
            try:
                audio = r.listen(source)
                text = r.recognize_google(audio, language='es-ES')
                print(text)
                text = text.split()
                if 'empezar' in text or 'comenzar' in text:
                    print("Game is runing")
                    accio = "a1"

                elif 'terminar' in text or 'parar' in text:
                        print("Game finished")
                        accio = "a2"

                elif 'reiniciar' in text or 'reinicia' in text:
                    accio = "a4"

                elif 'cambiar' in text:
                    accio = "a5"
                    self.voice_solving = not self.voice_solving
                    self.dir = (0, 0)

                elif self.voice_solving:
                    p = self.dir[0]
                    r = self.dir[1]
                    if 'arriba' in text:
                        accio = "a6"
                        p = -1
                    if 'abajo' in text:
                        accio = "a6"
                        p = 1
                    if 'izquierda' in text:
                        accio = "a6"
                        r = -1
                    if 'derecha' in text:
                        accio = "a6"
                        r = 1
                    if 'recto' in text:
                        accio = "a6"
                        p = 0
                        r = 0

                    self.dir = (p, r)
                    coord = self.dir

                else:
                     print("No hay comando asociado a esta palabra")

            except sr.UnknownValueError:
                print('Error, no se entendio la palabra')
                return accio, coord
            except sr.RequestError as e:
                print('failed'.format(e))
                return accio, coord
        return accio,coord



