import cv2
import numpy as np

# Abre la cámara
cap = cv2.VideoCapture(0)

# Cargamos una imagen de referencia para comparar
reference_image = cv2.imread('axel.png')
reference_encoding = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Crea un objeto detector de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Lee un frame de la cámara
    ret, frame = cap.read()

    # Convierte el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta caras en el frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Recorta la región de la cara
        face = gray[y:y+h, x:x+w]

        # Compara la cara detectada con la imagen de referencia
        result = cv2.matchTemplate(face, reference_encoding, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7  # Umbral de similitud

        if np.max(result) > threshold:
            name = "Persona de referencia"
        else:
            name = "Desconocido"

        # Dibuja un rectángulo alrededor de la cara y muestra el nombre
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Muestra el frame con las caras detectadas
    cv2.imshow('Reconocimiento Facial', frame)

    # Sale del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos y cierra la cámara
cap.release()
cv2.destroyAllWindows()
