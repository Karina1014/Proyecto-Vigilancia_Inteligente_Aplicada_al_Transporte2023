#Librerias
from FlaskProyecto import app 
import cv2 #Visión de imagen y video
import winsound #Biblioteca en windows que se utiliza para reproducir sonido
import os #Proporciona una serie de funciones para interactuar con el sistema operativo
import mediapipe as mp #Modelo face_mesh
import numpy as np #Funciones matemáticas
import matplotlib.pyplot as plt #Para la visualización de datos y creación de gráficos.
from collections import deque #Representa una estructura de datos

#Dependencia para validar credenciales de Ingreso
from flask import Flask, render_template, Response, url_for, redirect, request, flash #Herramientaspara Desarrollo web /solicitudes http y direcciones html
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
 

#Dirijo a la raíz del sitio web 
@app.route('/')
def index1():
    return render_template('Principal.html')
#Dirijo para la nueva ruta del Login
@app.route('/Usuario')
def nueva_ruta1():
    return render_template('Login.html')

#Dirijo a la ruta definida
@app.route('/redirigir')
def redirigir1():
    return redirect(url_for('nueva_ruta1'))

#Dirijo para la nueva ruta del proyecto IA
@app.route('/nueva_ruta/proyecto')
def nueva_ruta():
    return render_template('indexPagina2.html')

#Dirijo a la ruta definida
@app.route('/nueva_ruta/redirigir')
def redirigir():
    return redirect(url_for('nueva_ruta'))


#Login Metodos de get y post "credenciales"--------------------------

#Inicio de control de ingreso se credenciales
app.secret_key = 'your_secret_key'
login_manager = LoginManager(app)
login_manager.login_view = 'login'


#Diccionario para almacenar las credenciales de los usuarios
users = {
    'grupo5': {'password': 'grupo5'},
    'admin': {'password': 'admin'},
}

# Clase para representar al usuario
class User(UserMixin):
    def __init__(self, username):
        self.id = username
       # Aquí podrías cargar más información del usuario desde la base de datos, por ejemplo.

# Función para cargar el usuario en la sesión
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Ruta para el inicio de sesión
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and user['password'] == password:
            # Marcar al usuario como autenticado
            login_user(User(username))
            return redirect(url_for('nueva_ruta'))
        else:
            flash('Usuario o contrasena incorrectos', 'error')
    return render_template('login.html')

# Ruta para cerrar sesión
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

#Fin de control de ingreso se credenciales--------------------------



# Cargar el clasificador Haar Cascade para la detección de ojos de OpenCV
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter):
    # Crear una imagen auxiliar en blanco con las mismas dimensiones que el marco de entrada (frame).
    aux_image = np.zeros(frame.shape, np.uint8)

    # Crear arreglos NumPy con las coordenadas del ojo izquierdo y derecho.
    contours1 = np.array([coordinates_left_eye])
    contours2 = np.array([coordinates_right_eye])

    # Rellenar el contorno del ojo izquierdo en la imagen auxiliar con el color azul.
    cv2.fillPoly(aux_image, pts=[contours1], color=(255, 0, 0))

    # Rellenar el contorno del ojo derecho en la imagen auxiliar con el color azul.
    cv2.fillPoly(aux_image, pts=[contours2], color=(255, 0, 0))

    # Combinar la imagen original (frame) con la imagen auxiliar que contiene los ojos resaltados.
    # El parámetro 0.7 controla la transparencia del resaltado de ojos en la imagen de salida (output).
    output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1)

    # Dibujar un rectángulo relleno en la parte superior izquierda de la imagen de salida,
    # que actuará como una barra para mostrar el número de parpadeos.
    cv2.rectangle(output, (0, 0), (200, 50), (255, 0, 0), -1)

    # Dibujar un rectángulo sin relleno en la parte superior derecha de la imagen de salida,
    # alrededor del área donde se mostrará el número de parpadeos.
    cv2.rectangle(output, (202, 0), (265, 50), (255, 0, 0), 2)

    # Agregar un texto en la imagen de salida que indica "Num. Parpadeos:" en la barra creada.
    cv2.putText(output, "Num. Parpadeos:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Agregar el número de parpadeos (blink_counter) en la parte derecha de la barra.
    cv2.putText(output, "{}".format(blink_counter), (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 0, 250), 2)

    # Devolver la imagen de salida con los ojos resaltados y la barra que muestra el número de parpadeos.
    return output


def eye_aspect_ratio(coordinates):
    # Calcular la distancia euclidiana entre el punto 1 (coordenadas[1]) y el punto 5 (coordenadas[5]).
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))

    # Calcular la distancia euclidiana entre el punto 2 (coordenadas[2]) y el punto 4 (coordenadas[4]).
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))

    # Calcular la distancia euclidiana entre el punto 0 (coordenadas[0]) y el punto 3 (coordenadas[3]).
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))

    # Calcular el índice de relación de aspecto del ojo.
    # El índice de relación de aspecto se calcula como (d_A + d_B) / (2 * d_C).
    return (d_A + d_B) / (2 * d_C)



def plotting_ear(pts_ear, line1):
    # Declaramos 'figure' como una variable global para poder acceder y modificar fuera de la función
    global figure
    
    # Generamos un array de 64 puntos equidistantes entre 0 y 1
    pts = np.linspace(0, 1, 64)
    
    # Verificamos si 'line1' está vacío (lista vacía)
    if line1 == []:
        # Establecemos el estilo de la gráfica como 'ggplot'
        plt.style.use("ggplot")
        
        # Activamos el modo interactivo de Matplotlib para actualizar la gráfica en tiempo real
        plt.ion()
        
        # Creamos una nueva figura y ejes para la gráfica
        figure, ax = plt.subplots()
        
        # Trazamos una línea con los puntos 'pts_ear' en el eje Y y 'pts' en el eje X
        line1, = ax.plot(pts, pts_ear)
        
        # Establecemos los límites del eje Y entre 0.1 y 0.4
        plt.ylim(0.1, 0.4)
        
        # Establecemos los límites del eje X entre 0 y 1
        plt.xlim(0, 1)
        
        # Establecemos el título del eje Y como "ojos" con un tamaño de fuente de 18
        plt.ylabel("Umbral", fontsize=18)
    else:
        # Si 'line1' no está vacío, actualizamos los datos de la línea con los nuevos valores 'pts_ear'
        line1.set_ydata(pts_ear)
        
        # Dibujamos la gráfica actualizada
        figure.canvas.draw()
        
        # Actualizamos la interfaz de la gráfica para mostrar los cambios en tiempo real
        figure.canvas.flush_events()

    # Devolvemos el objeto 'line1', que contiene la referencia a la línea trazada en la gráfica
    return line1



#Generar el frame con el vídeo en tiempo real, además de tener los puntos de referencia de cada ojo y contar los parpadeos
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #Detecta la cámara 0 si es por defecto 1 si es externa

    mp_face_mesh = mp.solutions.face_mesh #Importa la solución de facemesh de mediapipe para detectar landmarcks faciales
    index_left_eye = [33, 160, 158, 133, 153, 144] #Listado de landmarks(puntos de referencia) facilaes del ojo izquierdo, coordenadas relevantes que se van a utilizar
    index_right_eye = [362, 385, 387, 263, 373, 380] #Listado de landmarcks facilaes del ojo derecho, coordenadas relevantes que se van a utilizar
    EAR_THRESH = 0.26 #Umbral de los ojos abiertos
    NUM_FRAMES = 2 #Número de veces que se tiene para contabilizar que un ojo se cerro y abrio
    NUM_FRAMES_DORMIDO = 20 #Valor para verificar que tiene cerrado los ojos por más de 20 frames
    aux_counter = 0 #Seguimiento de frames para determinación de somnolencia
    blink_counter = 0 #Conteo de parpadeos
    umbral_superado_counter = 0 #Contador para realizar seguimiento de que se ha superado el umbral de ojos cerrados NUM_FRAMES_DORMIDO
    line1 = [] #Arreglo vacío 
    pts_ear = deque(maxlen=64) #Cola para guardar los datos de la lectura del umbral en tiempo real
    i = 0

    #Permite reconocer un rostro para poder realizar el mallado con los puntos de referencia
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,  #Se pone parámetro false cuando se trabaja con vídeo en streaming y True cuando se trabaja con imágenes
            max_num_faces=1) as face_mesh: 

        #Dentro del while se va a leer la imagen en tiempo de ejecución y se detectará el parpadeo de la persona
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.flip(frame, 1) #cv2.flip permite dar la vuelta a la imagen para verla en forma de espejo
            height, width, _ = frame.shape #Obtenemos la dimesión del frame para poder procesar los fotogramas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            coordinates_left_eye = [] #Lista vacía para guardar los puntos de referencia del ojo izquierdo
            coordinates_right_eye = [] #Lista vacía para guardar los puntos de referencia del ojo izquierdo

            
            #Permite detectar que exista un rostro y poder asignar los landmarks de referencia para el rostro          
            if results.multi_face_landmarks is not None:
                for face_landmarks in results.multi_face_landmarks:
                    for index in index_left_eye:  #Permite asignar los puntos de referencia para el ojo izquierdo
                        x = int(face_landmarks.landmark[index].x * width)
                        y = int(face_landmarks.landmark[index].y * height)
                        coordinates_left_eye.append([x, y])
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                        cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
                    for index in index_right_eye: #Permite asignar los puntos de referencia para el ojo derecho
                        x = int(face_landmarks.landmark[index].x * width)
                        y = int(face_landmarks.landmark[index].y * height)
                        coordinates_right_eye.append([x, y])
                        cv2.circle(frame, (x, y), 2, (128, 0, 250), 1)
                        cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
                #Permite realizar el cálculo del índice de apertura ocular (Eye Aspect Ratio) para poder comparar con el umbral de 0.26 que definimos
                ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
                ear_right_eye = eye_aspect_ratio(coordinates_right_eye)
                ear = (ear_left_eye + ear_right_eye) / 2 


                # Ojos cerrados           
                if ear < EAR_THRESH:
                    aux_counter += 1
                    umbral_superado_counter += 1
                    if umbral_superado_counter >= NUM_FRAMES_DORMIDO:      #Permite determinar cuando el usuario mantiene cerrado los ojos por más de 20 iteracciones                  
                         # Mostrar mensaje en la imagen  
                        aux_image = np.zeros(frame.shape, np.uint8)                      
                        output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1)
                        cv2.rectangle(frame, (115, 410), (560, 460), (0, 0, 255), -1)
                        cv2.putText(frame, "Estas quedandote dormido", (130, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)                                                 
                        frecuencia = 1000  # Frecuencia en Hz
                        duracion_ms = 100  # Duración del pitido en milisegundos
                        winsound.Beep(frecuencia, duracion_ms)
                else:
                     if aux_counter >= NUM_FRAMES:  #Si el aux_counter es mayor a 2 quiere decir que se ha realizado un parpadeo y se contará
                         aux_counter = 0
                         blink_counter += 1
                     umbral_superado_counter = 0


                frame = drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter) #Dibuja el frame, los puntos en los ojos y el conteo
                pts_ear.append(ear) #Actualiza el valor del deque con el valor reciente del índice de apertura ocular para visualizarlo en pantalla
                #Asegura que en la colección pts_ear se hayan guardado 64 datos antes de empezar a mostrar la gráfica y comprueba que el reconocimiento del rostro es el correcto
                if i > 70:
                    line1 = plotting_ear(pts_ear, line1) #Permite guardar los datos del conteo de los ojos para ponerlos en la gráfica una vez que se hayan realizado 70 interaciones
                i += 1

            # Los fotogramas codificados los pasa a formato jpg y los trasnforma a bytes para visualizarlos en frame_procesado
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_procesado = jpeg.tobytes()

            #Visualiza los fotogramas en el frame durante la ejecución
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_procesado + b'\r\n\r\n')
    #Libera los recursos utilizados para la lectura de vídeo por cámara
    cap.release()


#Ruta video camara
@app.route('/video_feed')
def videoCamara():
    # Retorna el contenido de la transmisión en tiempo real
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#Ruta de Proyecto VIAT
@app.route('/Proyecto VIAT')
def index():
    return render_template('indexPagina2.html')

#Puerto por Defecto
if __name__ == '__main__':
    app.run(debug=True)

 #if name == 'main':
 #app.run(debug=True, port=9000)

