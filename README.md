# PROJECTE RLP CURS 2019 - 2020 -- LASO
<img src="https://github.com/MarcMDE/LASO/blob/master/SimulacioRLP_py/src/frame_tests/f0.jpg?raw=true" align="right" width="300" alt="header pic"/>
# PARTICIPANTS

* Narcís Nogué Bonet - 1493389
* Raquel Navarro Ballesteros - 1423208
* Marc Montagut Llauradó - 1462234
* Aleix Torres Rodriguez - 1493910
* Aleix Saus Mestres - 1458353



# REQUERIMENTS

Es troben també al fitxer requirements.txt per instalar amb pip install

* cachetools==4.1.0
* certifi==2020.4.5.1
* chardet==3.0.4
* google-api-core==1.17.0
* google-api-python-client==1.8.3
* google-auth==1.15.0
* google-auth-httplib2==0.0.3
* google-cloud==0.34.0
* google-cloud-core==1.3.0
* google-cloud-speech==1.3.2
* google-cloud-storage==1.28.1
* google-resumable-media==0.5.0
* googleapis-common-protos==1.51.0
* grpcio==1.29.0
* httplib2==0.18.1
* idna==2.9
* numpy==1.18.3
* opencv-python==4.2.0.34
* panda3d==1.10.6
* protobuf==3.12.1
* pyasn1==0.4.8
* pyasn1-modules==0.2.8
* PyAudio==0.2.11
* pytz==2020.1
* requests==2.23.0
* rsa==4.0
* six==1.15.0
* SpeechRecognition==3.8.1
* uritemplate==3.0.1
* urllib3==1.25.9

# COM UTILITZAR

Moviment de camera:
	-Botó dret - Zoom
	-Botó esquerre - Moviment lateral i vertical
	-Botó central - Rotar entorn al centre

Iniciar resolució - "1"

Moure laberint manualment - "Up", "Down", "Left", "Right"
	(Assegurar-se que la pantalla amb la càmera principal està sel·leccionada)	

Moure làmpara per cambiar la llum - "j", "k", "l", "i"

Cambiar intensitat llum global - "8", "9"

Ressetejar posició càmera - "r"


Comandes de veu:
	empezar / comenzar - Activar la simulació
	para / parar - Parar simulació
	reiniciar / reinicia - Tornar a començar la resolució
	comandos / comando - Activar resolució per veu
	automático - Activar resolució automàtica

Comandes de veu en resolució per veu:
	arriba
	abajo
	izquierda
	derecha
	recto - Estabilitza la bola en la posició on es troba

# MODULS

1- Reconeixement de veu: Mitjançant un micròfon, el robot capta a temps real les ordres que l’usuari li digui i segons la comanda, farà una cosa o altre. Hem seleccionat aquest mòdul per donar-li un punt més interactiu amb l’usuari.

2- Reconeixement visual: Mitjançant una càmera, el robot captarà el laberint en “temps real”. A continuació, es buscaran els quatre cantons del laberint i es realitzarà una homografía per tal de corregir la perspectiva de l’imatge. Finalment, i mitjançant el color, s'identificaran els diferents elements del laberint (parets, forats i camins). No es descarta intentar implementar un reconeixement per forma (“shape recognition”) de forma posterior, en cas de tenir temps. 

3- Resolució de laberint: Un cop el robot té tota la informació del laberint en una imatge, aquest resol tot el laberint de cop, és a dir, troba tot el path que seguirà la bola. Per fer-ho, primer infla les parets del laberint pel valor del radi de la bola, de manera que assegura que el que queda d’espai lliure són punts on es podria col·locar el centre de la bola, i després infla els forats un cert nombre de píxels que serveixen de marge de seguretat. Tot el que queda llavors és espai per on es pot moure la bola.
Finalment el robot resol el laberint utilitzant l’algoritme A* i retorna una llista de punts (píxels) per on s’ha de moure la bola per anar des del punt inicial fins al punt final.
Un cop el robot té el laberint resolt utilitza un sistema de control PID per fer que la bola segueixi punt per punt el camí trobat en el punt anterior. El sistema PID agafa la distància entre la posició de la bola i el punt on ha d’estar actualment com a error i calcula l’angle en que s’han de col·locar els dos eixos mòbils del tauler com a resultat.

4- Control de servomotors: Aquest mòdul ”tradueix” l’angle del taulell a l’angle que han de col·locar-se els servomotors per a aconseguir que la pilota es mogui on volem.

5- Simulació: Es realitzarà una simulació funcional del robot en el game engine “Unity”. S’espera realitzar una implementació completa de tots els mòduls, per aquest motiu aquests es desenvoluparan tenint el compte l’entorn Unity.  

