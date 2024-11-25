package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

/**
 * A classe detector escapsula a lógica de carregamento do modelo, inferencia e pos-processamento.
 */
class Detector(
    private val context: Context,                 //contexto para recursos, como o arquivo de modelo e de labels
    private val modelPath: String,                 //caminho do proprio arquivo que armazena o modelo
    private val labelPath: String,                 //caminho dos labels do modelo
    private val detectorListener: DetectorListener //interface que permite realizar as predicoes
) {

    private var interpreter: Interpreter? = null   //Referência ao modelo carregado.
    private var labels = mutableListOf<String>()   //lista de labels do arquivo .txt

    // variaveis de dimensao do tensor de entrada, numero de classes e elementos (tensor de saida)
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    // configuracao de pre-processamento da imagem
    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))   //normaliza os pixels da imagem de entrada
        .add(CastOp(INPUT_IMAGE_TYPE))                            //converte para o tipo correto, um casting de tipo.
        .build()

    //Este metodo carrega o modelo
    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()
        options.numThreads = 4
        interpreter = Interpreter(model, options)
        // pegam o acesso a entrada e a saida do modelo
        // o operador de coalescência ?. é um SAVE CALL OPERATOR, verifica se o objeto precede o operador é NULL, se for nulo o metodo nao é chamado, esta relacionado ao conceito de nulidade de kotlin
        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return
        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]
        numChannel = outputShape[1]
        numElements = outputShape[2]

        try {
            val inputStream: InputStream = context.assets.open(labelPath) //abre o arquivo de labels em um formato que permite ler em uma sequencia de bytes
            val reader = BufferedReader(InputStreamReader(inputStream)) //converte o fluxo de bytes de inputStream em um fluxo de caracteres

            var line: String? = reader.readLine()//le a proxima linha do arquivo, retornando string ou null
            while (line != null && line != "") { //le enquanto o arquivo nao é null nem vazio
                labels.add(line) //adiciona os rotulos das classes do txt
                line = reader.readLine()
            }
            // depois que acabar fecha esse processo de leitura dos rotulos disponiveis
            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    //limpa os recursos que o mudelo ja usou
    fun clear() {
        interpreter?.close()
        interpreter = null
    }

    fun detect(frame: Bitmap) {
        interpreter ?: return //verifica se o objeto interpreter esta disponivel
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return

        var inferenceTime = SystemClock.uptimeMillis() //retorna o tempo de processamento (inferencia em ms)

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false) //redimensiona a imagem para o padrao da entrada do modelo. Sem usar filtros.

        val tensorImage = TensorImage(DataType.FLOAT32) //cria um objeto que suporta o formato de float32 o tipo esperado pelo modelo
        tensorImage.load(resizedBitmap) //carrega a imagem redimensionada
        val processedImage = imageProcessor.process(tensorImage) //faz o pre processamento da imagem e a armazena
        val imageBuffer = processedImage.buffer //extrai o buffer de dados da imagem processada, contendo pixels em formato de ponto flutuante, adequados para o modelo.

        val output = TensorBuffer.createFixedSize(intArrayOf(1 , numChannel, numElements), OUTPUT_IMAGE_TYPE) //cria o tensor de saida com os canais de saida(deteccoes, classes etc.)
        interpreter?.run(imageBuffer, output.buffer) // Executa o modelo com os dados de entrada da var imageBuffer, e armazena a saida em output.buffer


        val bestBoxes = bestBox(output.floatArray) //Chamada da funcao bestBox que calcula as melhores caixas delimitadoras para um objeto, similar ao NMS.
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime //calcula o tempo de inferencia


        if (bestBoxes == null) {
            detectorListener.onEmptyDetect() //se nenhum objeto for detectado
            return
        }

        detectorListener.onDetect(bestBoxes, inferenceTime) // caso seja detectado
    }

    // essa funcao cria as caixas delimitadoras para cada objeto
    private fun bestBox(array: FloatArray) : List<BoundingBox>? { //Espera a entrada do modelo, contendo as caixas e probabilidade de cada classe prevista, e retorna a lista de caixas delimitadoras

        val boundingBoxes = mutableListOf<BoundingBox>() //lista de caixas delimitadoras

        for (c in 0 until numElements) { //cada c representa uma possivle deteccao
            //faz a busca da classe com maior confiança
            var maxConf = -1.0f
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel){
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) { //filtro de confiança, ignora detecoes com baixa confiança
                //calculo das Coordenadas da caixa
                val clsName = labels[maxIdx]
                val cx = array[c] // 0
                val cy = array[c + numElements] // 1
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                //faz o calculo do limite da caixa a ser desenhada
                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes) //Após processar todas as detecções, aplica supressão de não máximos para remover caixas redundantes.
    }

    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList() //Ordena as caixas pela confiança (cnf) em ordem decrescente.
        val selectedBoxes = mutableListOf<BoundingBox>()

        while(sortedBoxes.isNotEmpty()) {
            //Seleciona a caixa com maior confiança (first) e a remove da lista.
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) { //Para cada caixa restante, calcula a Interseção sobre União (IoU)
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) { //remove caixas com iou superior a constante de iou
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    //calcula a intersecao sobre uniao (IoU)
    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    interface DetectorListener { //permite definicao de contrato para permitir invocao
        fun onEmptyDetect() //quando nenhuma deteccao for encontrada
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) //Método chamado quando detecções são encontradas
    }

    // Companion object é usado para agrupar constantes e valores relacionados que são compartilhados por todas as instâncias da classe.
    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
    }
}