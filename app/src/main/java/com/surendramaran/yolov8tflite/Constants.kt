package com.surendramaran.yolov8tflite

// Define um objeto SINGLETON (Classe que possui uma uníca instancia) chamado Constants, ele é usado para armazenar constantes
// com o path do modelo e do arquivo que contem as classes

object Constants {
    const val MODEL_PATH = "model.tflite"
    const val LABELS_PATH = "labels.txt"
}
