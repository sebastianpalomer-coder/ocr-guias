import io
import os
import re
import tempfile
import unicodedata

import cv2
import numpy as np
import pytesseract
import pypdfium2 as pdfium

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageOps
from pytesseract import Output


app = FastAPI()


FORMATOS_IMAGEN = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}

FORMATO_PDF = "application/pdf"


# ============================================================
# API
# ============================================================

@app.get("/ping")
def ping():

    return {
        "status": "ok",
        "service": "ocr-guias",
        "version": "2.2-transex"
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):

    try:

        datos = await file.read()

        if not datos:

            raise HTTPException(
                status_code=400,
                detail="Archivo vacío"
            )

        tipo = file.content_type or ""

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if tipo == FORMATO_PDF:

            resultado = procesar_pdf(datos)

        # ----------------------------------------------------
        # JPG / PNG
        # ----------------------------------------------------

        elif tipo in FORMATOS_IMAGEN:

            imagen = Image.open(
                io.BytesIO(datos)
            )

            imagen = ImageOps.exif_transpose(
                imagen
            )

            resultado = procesar_imagen(
                imagen
            )

            resultado["pages"] = 1

        else:

            raise HTTPException(
                status_code=400,
                detail=f"Formato no soportado: {tipo}"
            )

        resultado.update({
            "ok": True,
            "filename": file.filename,
            "content_type": tipo
        })

        return resultado


    except HTTPException:

        raise


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# PDF
# ============================================================

def procesar_pdf(datos: bytes):

    textos = []
    documentos = []
    detalles = []
    tablas = []

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as tmp:

        tmp.write(datos)

        ruta_pdf = tmp.name

    try:

        pdf = pdfium.PdfDocument(
            ruta_pdf
        )

        total_paginas = len(pdf)

        for numero_pagina in range(
            total_paginas
        ):

            pagina = pdf[
                numero_pagina
            ]

            bitmap = pagina.render(
                scale=3
            )

            imagen = bitmap.to_pil()

            resultado_pagina = (
                procesar_imagen(
                    imagen
                )
            )

            textos.append(
                "--- PAGINA {} ---\n{}".format(
                    numero_pagina + 1,
                    resultado_pagina[
                        "text"
                    ]
                )
            )

            documentos.append(
                resultado_pagina[
                    "documento"
                ]
            )

            detalles.extend(
                resultado_pagina[
                    "detalle"
                ]
            )

            tablas.append(
                resultado_pagina.get(
                    "tabla_texto",
                    ""
                )
            )

        documento = (
            documentos[0]
            if documentos
            else {}
        )

        return {

            "pages":
                total_paginas,

            "text":
                "\n\n".join(
                    textos
                ),

            "documento":
                documento,

            "detalle":
                detalles,

            "tabla_texto":
                "\n\n".join(
                    tablas
                )
        }


    finally:

        try:

            os.remove(
                ruta_pdf
            )

        except Exception:

            pass


# ============================================================
# PROCESAR IMAGEN
# ============================================================

def procesar_imagen(
    imagen: Image.Image
):

    # --------------------------------------------------------
    # Orientación
    # --------------------------------------------------------

    imagen = ImageOps.exif_transpose(
        imagen
    )

    # --------------------------------------------------------
    # Corrección perspectiva
    # --------------------------------------------------------

    imagen_documento = (
        corregir_perspectiva(
            imagen
        )
    )

    # --------------------------------------------------------
    # Preparar para OCR
    # --------------------------------------------------------

    imagen_preparada = (
        preparar_imagen(
            imagen_documento
        )
    )

    # --------------------------------------------------------
    # OCR página completa
    # --------------------------------------------------------

    texto_completo = (
        ejecutar_ocr_texto(
            imagen_preparada,
            psm=6
        )
    )

    # --------------------------------------------------------
    # Datos generales
    # --------------------------------------------------------

    documento = (
        extraer_documento(
            texto_completo
        )
    )

    # --------------------------------------------------------
    # PRODUCTOS
    #
    # IMPORTANTE:
    #
    # Desde versión 2.2 usamos principalmente
    # el OCR COMPLETO, porque resultó más confiable
    # que volver a hacer OCR sobre el recorte.
    # --------------------------------------------------------

    detalle = (
        interpretar_productos_transex(
            texto_completo
        )
    )

    # --------------------------------------------------------
    # TABLA OCR
    #
    # Se conserva solamente para auditoría.
    # NO se utiliza como fuente principal.
    # --------------------------------------------------------

    datos_posiciones = (
        ejecutar_ocr_datos(
            imagen_preparada,
            psm=6
        )
    )

    tabla = (
        recortar_zona_productos(
            imagen_preparada,
            datos_posiciones
        )
    )

    tabla_texto = ""

    if tabla is not None:

        tabla = preparar_recorte(
            tabla
        )

        tabla_texto = (
            ejecutar_ocr_texto(
                tabla,
                psm=6
            )
        )

    return {

        "text":
            texto_completo,

        "documento":
            documento,

        "detalle":
            detalle,

        "tabla_texto":
            tabla_texto
    }


# ============================================================
# PREPARAR IMAGEN
# ============================================================

def preparar_imagen(
    imagen: Image.Image
):

    imagen = imagen.convert(
        "L"
    )

    imagen = ImageOps.autocontrast(
        imagen
    )

    ancho, alto = imagen.size

    ancho_objetivo = 2200

    if ancho < ancho_objetivo:

        factor = (
            ancho_objetivo
            / ancho
        )

        imagen = imagen.resize(

            (
                int(
                    ancho * factor
                ),

                int(
                    alto * factor
                )
            ),

            Image.Resampling.LANCZOS
        )

    return imagen


# ============================================================
# PREPARAR RECORTE TABLA
# ============================================================

def preparar_recorte(
    imagen: Image.Image
):

    """
    No eliminamos líneas.

    En la versión anterior la eliminación
    de líneas estaba borrando partes de
    números como 83.778 y 251.334.
    """

    imagen = imagen.convert(
        "L"
    )

    imagen = ImageOps.autocontrast(
        imagen
    )

    ancho, alto = imagen.size

    imagen = imagen.resize(

        (
            int(
                ancho * 1.5
            ),

            int(
                alto * 1.5
            )
        ),

        Image.Resampling.LANCZOS
    )

    return imagen


# ============================================================
# CORRECCIÓN PERSPECTIVA
# ============================================================

def corregir_perspectiva(
    imagen: Image.Image
):

    rgb = np.array(
        imagen.convert(
            "RGB"
        )
    )

    original = rgb.copy()

    alto, ancho = rgb.shape[:2]

    escala = (
        1200
        / max(
            alto,
            ancho
        )
    )

    if escala < 1:

        pequena = cv2.resize(

            rgb,

            None,

            fx=escala,

            fy=escala
        )

    else:

        pequena = rgb.copy()

        escala = 1

    gris = cv2.cvtColor(
        pequena,
        cv2.COLOR_RGB2GRAY
    )

    gris = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    bordes = cv2.Canny(
        gris,
        50,
        150
    )

    bordes = cv2.dilate(
        bordes,
        None,
        iterations=1
    )

    contornos, _ = cv2.findContours(

        bordes,

        cv2.RETR_LIST,

        cv2.CHAIN_APPROX_SIMPLE
    )

    contornos = sorted(

        contornos,

        key=cv2.contourArea,

        reverse=True

    )[:10]

    pagina = None

    area_imagen = (
        pequena.shape[0]
        * pequena.shape[1]
    )

    for contorno in contornos:

        perimetro = (
            cv2.arcLength(
                contorno,
                True
            )
        )

        aproximado = (
            cv2.approxPolyDP(
                contorno,
                0.02 * perimetro,
                True
            )
        )

        if len(
            aproximado
        ) != 4:

            continue

        area = cv2.contourArea(
            aproximado
        )

        if (
            area
            < area_imagen * 0.45
        ):

            continue

        pagina = aproximado.reshape(
            4,
            2
        ).astype(
            np.float32
        )

        break

    if pagina is None:

        return imagen

    pagina = (
        pagina
        / escala
    )

    ordenados = ordenar_puntos(
        pagina
    )

    tl, tr, br, bl = ordenados

    ancho_superior = (
        np.linalg.norm(
            tr - tl
        )
    )

    ancho_inferior = (
        np.linalg.norm(
            br - bl
        )
    )

    ancho_final = int(
        max(
            ancho_superior,
            ancho_inferior
        )
    )

    alto_izquierdo = (
        np.linalg.norm(
            bl - tl
        )
    )

    alto_derecho = (
        np.linalg.norm(
            br - tr
        )
    )

    alto_final = int(
        max(
            alto_izquierdo,
            alto_derecho
        )
    )

    if (
        ancho_final < 500
        or
        alto_final < 500
    ):

        return imagen

    destino = np.array(

        [

            [0, 0],

            [
                ancho_final - 1,
                0
            ],

            [
                ancho_final - 1,
                alto_final - 1
            ],

            [
                0,
                alto_final - 1
            ]

        ],

        dtype=np.float32
    )

    matriz = (
        cv2.getPerspectiveTransform(
            ordenados,
            destino
        )
    )

    corregida = cv2.warpPerspective(

        original,

        matriz,

        (
            ancho_final,
            alto_final
        )
    )

    return Image.fromarray(
        corregida
    )


def ordenar_puntos(
    puntos
):

    rect = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    suma = puntos.sum(
        axis=1
    )

    rect[0] = puntos[
        np.argmin(
            suma
        )
    ]

    rect[2] = puntos[
        np.argmax(
            suma
        )
    ]

    diferencia = np.diff(
        puntos,
        axis=1
    ).flatten()

    rect[1] = puntos[
        np.argmin(
            diferencia
        )
    ]

    rect[3] = puntos[
        np.argmax(
            diferencia
        )
    ]

    return rect


# ============================================================
# TESSERACT TEXTO
# ============================================================

def ejecutar_ocr_texto(
    imagen: Image.Image,
    psm=6
):

    config = (
        f"--oem 3 --psm {psm}"
    )

    texto = (
        pytesseract.image_to_string(

            imagen,

            lang="spa+eng",

            config=config
        )
    )

    return texto.strip()


# ============================================================
# TESSERACT COORDENADAS
# ============================================================

def ejecutar_ocr_datos(
    imagen: Image.Image,
    psm=6
):

    config = (
        f"--oem 3 --psm {psm}"
    )

    datos = (
        pytesseract.image_to_data(

            imagen,

            lang="spa+eng",

            config=config,

            output_type=Output.DICT
        )
    )

    palabras = []

    cantidad = len(
        datos["text"]
    )

    for i in range(
        cantidad
    ):

        texto = str(
            datos["text"][i]
        ).strip()

        if not texto:

            continue

        try:

            confianza = float(
                datos["conf"][i]
            )

        except Exception:

            confianza = -1

        palabras.append({

            "text":
                texto,

            "left":
                int(
                    datos["left"][i]
                ),

            "top":
                int(
                    datos["top"][i]
                ),

            "width":
                int(
                    datos["width"][i]
                ),

            "height":
                int(
                    datos["height"][i]
                ),

            "conf":
                confianza
        })

    return palabras


# ============================================================
# DATOS GENERALES
# ============================================================

def extraer_documento(
    texto
):

    texto_upper = texto.upper()

    documento = {

        "folio_ocr":
            None,

        "obra":
            None,

        "patente":
            None

    }

    # --------------------------------------------------------
    # FOLIO
    # --------------------------------------------------------

    resultado = re.search(
        r"N[°º]?\s*(\d{6,8})",
        texto_upper
    )

    if resultado:

        documento[
            "folio_ocr"
        ] = resultado.group(
            1
        )

    # --------------------------------------------------------
    # PATENTE
    # --------------------------------------------------------

    resultado = re.search(

        r"PATENTE\s*[:\-]?\s*"
        r"([A-Z]{2,4}[\-\s]?\d{2,4})",

        texto_upper
    )

    if resultado:

        patente = (
            resultado.group(1)
            .replace(
                " ",
                "-"
            )
        )

        documento[
            "patente"
        ] = patente

    # --------------------------------------------------------
    # OBRA
    # --------------------------------------------------------

    resultado = re.search(

        r"OBRA\s*[:|*]?\s*"
        r"([A-ZÁÉÍÓÚÑ0-9 \-]{3,40})",

        texto_upper
    )

    if resultado:

        obra = resultado.group(
            1
        )

        obra = re.split(
            r"[\n\r|\\]",
            obra
        )[0]

        obra = re.sub(
            r"\s+",
            " ",
            obra
        )

        documento[
            "obra"
        ] = obra.strip()

    return documento


# ============================================================
# PARSER PRINCIPAL PRODUCTOS TRANSEX
# ============================================================

def interpretar_productos_transex(
    texto
):

    """
    Busca directamente las líneas de productos
    dentro del OCR completo.

    Ejemplo real:

    4489 HORMIGON GR20-90%-40 C/08 D (e DN 3,00 83.778 ET

    CAR-INCOMP CARGA INCOMPLETA DE 7 Sp] MA] 3.00 18.390 a
    """

    resultados = []

    lineas = texto.splitlines()

    for linea_original in lineas:

        linea = re.sub(
            r"\s+",
            " ",
            linea_original
        ).strip()

        if not linea:

            continue

        linea_upper = (
            linea.upper()
        )

        # ----------------------------------------------------
        # HORMIGÓN
        # ----------------------------------------------------

        if (
            "HORMIGON"
            in linea_upper
        ):

            item = (
                interpretar_linea_hormigon(
                    linea
                )
            )

            if item:

                resultados.append(
                    item
                )

            continue

        # ----------------------------------------------------
        # CARGA INCOMPLETA
        # ----------------------------------------------------

        if (
            "CARGA"
            in linea_upper
            and
            "INCOMPLETA"
            in linea_upper
        ):

            item = (
                interpretar_linea_carga_incompleta(
                    linea
                )
            )

            if item:

                resultados.append(
                    item
                )

    return resultados


# ============================================================
# LÍNEA HORMIGÓN
# ============================================================

def interpretar_linea_hormigon(
    linea
):

    linea_upper = (
        linea.upper()
    )

    # --------------------------------------------------------
    # Código
    # --------------------------------------------------------

    resultado_codigo = re.match(
        r"^\s*(\d{3,7})\b",
        linea_upper
    )

    if not resultado_codigo:

        return None

    codigo = resultado_codigo.group(
        1
    )

    # --------------------------------------------------------
    # Descripción
    #
    # Ejemplo:
    #
    # HORMIGON GR20-90%-40 C/08
    # --------------------------------------------------------

    resultado_descripcion = re.search(

        r"(HORMIGON\s+.*?C/\d{1,3})",

        linea_upper
    )

    if not resultado_descripcion:

        return None

    descripcion = (
        resultado_descripcion
        .group(1)
        .strip()
    )

    posicion_fin_descripcion = (
        resultado_descripcion.end()
    )

    cola = linea[
        posicion_fin_descripcion:
    ]

    # --------------------------------------------------------
    # Cantidad y precio.
    # --------------------------------------------------------

    valores = (
        extraer_cantidad_precio_total(
            cola
        )
    )

    if (
        valores["cantidad"]
        is None
        or
        valores["precio"]
        is None
    ):

        return None

    total_calculado = round(
        valores["cantidad"]
        * valores["precio"]
    )

    total_coincide = None

    if (
        valores["total_ocr"]
        is not None
    ):

        total_coincide = (
            valores["total_ocr"]
            == total_calculado
        )

    return {

        "codigo":
            codigo,

        "descripcion":
            descripcion,

        "unidad":
            "M3",

        "unidad_ocr":
            None,

        "cantidad_raw":
            valores[
                "cantidad_raw"
            ],

        "cantidad":
            valores[
                "cantidad"
            ],

        "precio_raw":
            valores[
                "precio_raw"
            ],

        "precio":
            valores[
                "precio"
            ],

        "total_raw":
            valores[
                "total_raw"
            ],

        "total_ocr":
            valores[
                "total_ocr"
            ],

        "total_calculado":
            total_calculado,

        "total_coincide":
            total_coincide,

        "confianza_parser":
            100,

        "fuente":
            "TEXTO_OCR",

        "fila_raw":
            linea
    }


# ============================================================
# CARGA INCOMPLETA
# ============================================================

def interpretar_linea_carga_incompleta(
    linea
):

    linea_upper = (
        linea.upper()
    )

    # --------------------------------------------------------
    # Canonizamos código.
    #
    # OCR puede devolver:
    #
    # CAR-INCOMP
    # CAR-NCOMP
    # --------------------------------------------------------

    codigo = "CAR-INCOMP"

    resultado_descripcion = re.search(

        r"CARGA\s+INCOMPLETA",

        linea_upper
    )

    if not resultado_descripcion:

        return None

    descripcion = (
        "CARGA INCOMPLETA"
    )

    posicion_fin_descripcion = (
        resultado_descripcion.end()
    )

    cola = linea[
        posicion_fin_descripcion:
    ]

    valores = (
        extraer_cantidad_precio_total(
            cola
        )
    )

    if (
        valores["cantidad"]
        is None
        or
        valores["precio"]
        is None
    ):

        return None

    total_calculado = round(
        valores["cantidad"]
        * valores["precio"]
    )

    total_coincide = None

    if (
        valores["total_ocr"]
        is not None
    ):

        total_coincide = (
            valores["total_ocr"]
            == total_calculado
        )

    return {

        "codigo":
            codigo,

        "descripcion":
            descripcion,

        "unidad":
            "M3",

        "unidad_ocr":
            None,

        "cantidad_raw":
            valores[
                "cantidad_raw"
            ],

        "cantidad":
            valores[
                "cantidad"
            ],

        "precio_raw":
            valores[
                "precio_raw"
            ],

        "precio":
            valores[
                "precio"
            ],

        "total_raw":
            valores[
                "total_raw"
            ],

        "total_ocr":
            valores[
                "total_ocr"
            ],

        "total_calculado":
            total_calculado,

        "total_coincide":
            total_coincide,

        "confianza_parser":
            100,

        "fuente":
            "TEXTO_OCR",

        "fila_raw":
            linea
    }


# ============================================================
# EXTRAER CANTIDAD / PRECIO / TOTAL
# ============================================================

def extraer_cantidad_precio_total(
    texto
):

    resultado = {

        "cantidad_raw":
            None,

        "cantidad":
            None,

        "precio_raw":
            None,

        "precio":
            None,

        "total_raw":
            None,

        "total_ocr":
            None
    }

    # --------------------------------------------------------
    # Cantidad
    #
    # Ej:
    #
    # 3,00
    # 3.00
    # 10,50
    # --------------------------------------------------------

    coincidencia_cantidad = re.search(

        r"\b(\d{1,3}[,.]\d{2})\b",

        texto
    )

    if not coincidencia_cantidad:

        return resultado

    cantidad_raw = (
        coincidencia_cantidad
        .group(1)
    )

    cantidad = (
        convertir_decimal(
            cantidad_raw
        )
    )

    resultado[
        "cantidad_raw"
    ] = cantidad_raw

    resultado[
        "cantidad"
    ] = cantidad

    # --------------------------------------------------------
    # Buscar precio DESPUÉS de cantidad.
    # --------------------------------------------------------

    despues_cantidad = texto[
        coincidencia_cantidad.end():
    ]

    # Precio normalmente:
    #
    # 83.778
    # 18.390
    # 100.000
    #
    coincidencia_precio = re.search(

        r"\b(\d{1,3}(?:\.\d{3})+)\b",

        despues_cantidad
    )

    if not coincidencia_precio:

        return resultado

    precio_raw = (
        coincidencia_precio
        .group(1)
    )

    precio = convertir_monto(
        precio_raw
    )

    resultado[
        "precio_raw"
    ] = precio_raw

    resultado[
        "precio"
    ] = precio

    # --------------------------------------------------------
    # Buscar eventual total después del precio.
    # --------------------------------------------------------

    despues_precio = despues_cantidad[
        coincidencia_precio.end():
    ]

    coincidencia_total = re.search(

        r"\b(\d{1,3}(?:[\.\s]\d{3})+)\b",

        despues_precio
    )

    if coincidencia_total:

        total_raw = (
            coincidencia_total
            .group(1)
        )

        resultado[
            "total_raw"
        ] = total_raw

        resultado[
            "total_ocr"
        ] = convertir_monto(
            total_raw
        )

    return resultado


# ============================================================
# NÚMEROS
# ============================================================

def convertir_decimal(
    texto
):

    if texto is None:

        return None

    valor = str(
        texto
    ).strip()

    valor = valor.replace(
        ",",
        "."
    )

    try:

        return float(
            valor
        )

    except Exception:

        return None


def convertir_monto(
    texto
):

    if texto is None:

        return None

    valor = re.sub(
        r"[^0-9]",
        "",
        str(
            texto
        )
    )

    if not valor:

        return None

    try:

        return int(
            valor
        )

    except Exception:

        return None


# ============================================================
# RECORTAR ZONA PRODUCTOS PARA AUDITORÍA
# ============================================================

def recortar_zona_productos(
    imagen,
    palabras
):

    ancho, alto = imagen.size

    posiciones_inicio = []

    posiciones_fin = []

    for palabra in palabras:

        texto = (
            normalizar_texto(
                palabra[
                    "text"
                ]
            )
        )

        # ----------------------------------------------------
        # Inicio aproximado.
        # ----------------------------------------------------

        if (
            texto == "OBSERVACIONES"
        ):

            posiciones_inicio.append(
                palabra["top"]
                + palabra["height"]
            )

        # ----------------------------------------------------
        # Fin aproximado.
        # ----------------------------------------------------

        if (
            texto == "ITEM"
            or
            texto == "SOLICITANTE"
        ):

            posiciones_fin.append(
                palabra["top"]
            )

    if posiciones_inicio:

        top = min(
            posiciones_inicio
        )

    else:

        top = int(
            alto * 0.38
        )

    candidatos_fin = [

        valor

        for valor in posiciones_fin

        if valor > top

    ]

    if candidatos_fin:

        bottom = min(
            candidatos_fin
        )

    else:

        bottom = int(
            alto * 0.58
        )

    top = max(
        0,
        int(
            top - alto * 0.01
        )
    )

    bottom = min(
        alto,
        int(
            bottom + alto * 0.01
        )
    )

    if bottom <= top:

        return None

    return imagen.crop(

        (
            0,
            top,
            ancho,
            bottom
        )
    )


# ============================================================
# NORMALIZAR TEXTO
# ============================================================

def normalizar_texto(
    texto
):

    texto = str(
        texto or ""
    ).upper()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(

        caracter

        for caracter in texto

        if unicodedata.category(
            caracter
        ) != "Mn"
    )

    texto = re.sub(
        r"[^A-Z0-9]",
        "",
        texto
    )

    return texto
