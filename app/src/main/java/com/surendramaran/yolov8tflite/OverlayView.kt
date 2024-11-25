package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import java.util.LinkedList
import kotlin.math.max

/**
 * Classe OverlayView é responsável por desenhar caixas delimitadoras (bounding boxes) e seus rótulos na tela.
 * Usada para exibir resultados visuais de uma detecção (ex.: objetos detectados em uma imagem).
 */
class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    // Lista que contém os resultados da detecção (bounding boxes).
    private var results = listOf<BoundingBox>()

    // Objetos Paint para personalizar o desenho.
    private var boxPaint = Paint() // Para desenhar as caixas.
    private var textBackgroundPaint = Paint() // Para o fundo do texto.
    private var textPaint = Paint() // Para o texto.

    // Retângulo para medir as dimensões do texto.
    private var bounds = Rect()

    // Bloco inicializador que configura os Paints.
    init {
        initPaints()
    }

    /**
     * Limpa os resultados e reinicializa as configurações dos Paints.
     */
    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate() // Solicita que a View seja redesenhada.
        initPaints()
    }

    /**
     * Configura os Paints usados para desenhar o overlay.
     */
    private fun initPaints() {
        // Configuração do fundo do texto.
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        // Configuração do texto.
        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        // Configuração da caixa delimitadora.
        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    /**
     * Sobrescreve o método draw para desenhar caixas delimitadoras e rótulos na tela.
     * @param canvas Canvas onde os desenhos serão feitos.
     */
    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // Itera por cada bounding box nos resultados.
        results.forEach {
            // Calcula as coordenadas da bounding box em relação ao tamanho da tela.
            val left = it.x1 * width
            val top = it.y1 * height
            val right = it.x2 * width
            val bottom = it.y2 * height

            // Desenha o retângulo da bounding box.
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // Obtém o nome da classe detectada para exibição.
            val drawableText = it.clsName

            // Mede as dimensões do texto.
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()

            // Desenha o fundo do texto.
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Desenha o texto sobre o fundo.
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }
    }

    /**
     * Atualiza os resultados e solicita que a View seja redesenhada.
     * @param boundingBoxes Lista de BoundingBox com os novos resultados.
     */
    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate() // Redesenha a View.
    }

    companion object {
        // Padding usado ao redor do texto da bounding box.
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}