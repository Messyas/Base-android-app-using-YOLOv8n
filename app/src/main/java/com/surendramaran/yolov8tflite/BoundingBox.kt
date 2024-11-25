package com.surendramaran.yolov8tflite

/**
 * Classe de dados para armazenar as informações de uma caixa delimitadora (bounding box)
 */

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val cnf: Float, //confiança
    val cls: Int, //numero do array de classes
    val clsName: String //nome da classe
)