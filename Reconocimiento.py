import cv2
import face_recognition

# Abre la cámara
cap = cv2.VideoCapture(0)

# Cargamos una imagen de referencia para comparar
reference_image = face_recognition.load_image_file('axel.png')
reference_encoding = face_recognitionpyth.face_encodings(reference_image)[0]

while True:
    # Lee un frame de la cámara
    ret, frame = cap.read()

    # Busca caras en el frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compara la cara detectada con la imagen de referencia
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        name = "Desconocido"

        if True in matches:
            name = "Persona de referencia"

        # Dibuja un rectángulo alrededor de la cara y muestra el nombre
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Muestra el frame con las caras detectadas
    cv2.imshow('Reconocimiento Facial', frame)

    # Sale del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos y cierra la cámara
cap.release()
cv2.destroyAllWindows()
