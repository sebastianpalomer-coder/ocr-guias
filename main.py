import io
import os
import re
import tempfile
import statistics
import unicodedata
from difflib import SequenceMatcher

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
        "version": "2.0-transex"
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

            imagen = ImageOps.exif_transpose(imagen)

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

        for numero_pagina in range(total_paginas):

            pagina = pdf[numero_pagina]

            # Buena resolución para OCR
            bitmap = pagina.render(
                scale=3
            )

            imagen = bitmap.to_pil()

            resultado_pagina = procesar_imagen(
                imagen
            )

            textos.append(
                "--- PAGINA {} ---\n{}".format(
                    numero_pagina + 1,
                    resultado_pagina["text"]
                )
            )

            documentos.append(
                resultado_pagina["documento"]
            )

            detalles.extend(
                resultado_pagina["detalle"]
            )

            tablas.append(
                resultado_pagina.get(
                    "tabla_texto",
                    ""
                )
            )

        # Para guía de una página usamos
        # documento de primera página.
        documento = (
            documentos[0]
            if documentos
            else {}
        )

        return {
            "pages": total_paginas,
            "text": "\n\n".join(textos),
            "documento": documento,
            "detalle": detalles,
            "tabla_texto": "\n\n".join(tablas)
        }

    finally:

        try:
            os.remove(ruta_pdf)
        except Exception:
            pass


# ============================================================
# PROCESAR IMAGEN
# ============================================================

def procesar_imagen(imagen: Image.Image):

    # --------------------------------------------------------
    # 1. Corregir orientación / perspectiva
    # --------------------------------------------------------

    imagen = ImageOps.exif_transpose(
        imagen
    )

    imagen_documento = corregir_perspectiva(
        imagen
    )

    # --------------------------------------------------------
    # 2. Preparar imagen completa
    # --------------------------------------------------------

    imagen_preparada = preparar_imagen(
        imagen_documento
    )

    # --------------------------------------------------------
    # 3. OCR completo
    # --------------------------------------------------------

    texto_completo = ejecutar_ocr_texto(
        imagen_preparada
    )

    # --------------------------------------------------------
    # 4. OCR con posiciones
    # --------------------------------------------------------

    datos_ocr = ejecutar_ocr_datos(
        imagen_preparada
    )

    # --------------------------------------------------------
    # 5. Datos generales
    # --------------------------------------------------------

    documento = extraer_documento(
        texto_completo
    )

    # --------------------------------------------------------
    # 6. Detectar zona tabla
    # --------------------------------------------------------

    tabla = detectar_y_recortar_tabla(
        imagen_preparada,
        datos_ocr
    )

    detalle = []

    tabla_texto = ""

    if tabla is not None:

        tabla_limpia = limpiar_lineas_tabla(
            tabla
        )

        tabla_texto = ejecutar_ocr_texto(
            tabla_limpia,
            psm=6
        )

        datos_tabla = ejecutar_ocr_datos(
            tabla_limpia,
            psm=6
        )

        detalle = interpretar_tabla_transex(
            datos_tabla,
            tabla_limpia.width
        )

    return {
        "text": texto_completo,
        "documento": documento,
        "detalle": detalle,
        "tabla_texto": tabla_texto
    }


# ============================================================
# PREPARACIÓN DE IMAGEN
# ============================================================

def preparar_imagen(
    imagen: Image.Image
) -> Image.Image:

    imagen = imagen.convert("L")

    imagen = ImageOps.autocontrast(
        imagen
    )

    ancho, alto = imagen.size

    # Tesseract funciona mejor con documento grande.
    ancho_objetivo = 2200

    if ancho < ancho_objetivo:

        factor = (
            ancho_objetivo / ancho
        )

        imagen = imagen.resize(
            (
                int(ancho * factor),
                int(alto * factor)
            ),
            Image.Resampling.LANCZOS
        )

    return imagen


# ============================================================
# CORRECCIÓN DE PERSPECTIVA
# ============================================================

def corregir_perspectiva(
    imagen: Image.Image
) -> Image.Image:

    """
    Intenta identificar los cuatro bordes
    de la hoja.

    Si no consigue una detección segura,
    devuelve la imagen original.
    """

    rgb = np.array(
        imagen.convert("RGB")
    )

    original = rgb.copy()

    alto, ancho = rgb.shape[:2]

    # Imagen reducida para detección
    escala = 1200 / max(alto, ancho)

    if escala < 1:

        pequeña = cv2.resize(
            rgb,
            None,
            fx=escala,
            fy=escala
        )

    else:
        pequeña = rgb.copy()
        escala = 1

    gris = cv2.cvtColor(
        pequeña,
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
        pequeña.shape[0]
        * pequeña.shape[1]
    )

    for contorno in contornos:

        perimetro = cv2.arcLength(
            contorno,
            True
        )

        aproximado = cv2.approxPolyDP(
            contorno,
            0.02 * perimetro,
            True
        )

        if len(aproximado) != 4:
            continue

        area = cv2.contourArea(
            aproximado
        )

        # Debe representar una parte importante
        # de la fotografía.
        if area < area_imagen * 0.45:
            continue

        pagina = aproximado.reshape(
            4,
            2
        ).astype(np.float32)

        break

    if pagina is None:
        return imagen

    # Volver coordenadas al tamaño original.
    pagina = pagina / escala

    ordenados = ordenar_puntos(
        pagina
    )

    tl, tr, br, bl = ordenados

    ancho_superior = np.linalg.norm(
        tr - tl
    )

    ancho_inferior = np.linalg.norm(
        br - bl
    )

    ancho_final = int(
        max(
            ancho_superior,
            ancho_inferior
        )
    )

    alto_izq = np.linalg.norm(
        bl - tl
    )

    alto_der = np.linalg.norm(
        br - tr
    )

    alto_final = int(
        max(
            alto_izq,
            alto_der
        )
    )

    if (
        ancho_final < 500
        or alto_final < 500
    ):
        return imagen

    destino = np.array(
        [
            [0, 0],
            [ancho_final - 1, 0],
            [
                ancho_final - 1,
                alto_final - 1
            ],
            [0, alto_final - 1]
        ],
        dtype=np.float32
    )

    matriz = cv2.getPerspectiveTransform(
        ordenados,
        destino
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


def ordenar_puntos(puntos):

    rect = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    suma = puntos.sum(axis=1)

    rect[0] = puntos[
        np.argmin(suma)
    ]

    rect[2] = puntos[
        np.argmax(suma)
    ]

    diferencia = np.diff(
        puntos,
        axis=1
    ).flatten()

    rect[1] = puntos[
        np.argmin(diferencia)
    ]

    rect[3] = puntos[
        np.argmax(diferencia)
    ]

    return rect


# ============================================================
# TESSERACT
# ============================================================

def ejecutar_ocr_texto(
    imagen: Image.Image,
    psm=6
):

    config = (
        f"--oem 3 --psm {psm}"
    )

    texto = pytesseract.image_to_string(
        imagen,
        lang="spa+eng",
        config=config
    )

    return texto.strip()


def ejecutar_ocr_datos(
    imagen: Image.Image,
    psm=6
):

    config = (
        f"--oem 3 --psm {psm}"
    )

    datos = pytesseract.image_to_data(
        imagen,
        lang="spa+eng",
        config=config,
        output_type=Output.DICT
    )

    palabras = []

    cantidad = len(
        datos["text"]
    )

    for i in range(cantidad):

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
            "text": texto,
            "normalizado":
                normalizar_texto(texto),

            "left":
                int(datos["left"][i]),

            "top":
                int(datos["top"][i]),

            "width":
                int(datos["width"][i]),

            "height":
                int(datos["height"][i]),

            "conf":
                confianza
        })

    return palabras


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalizar_texto(texto):

    texto = str(
        texto or ""
    ).upper()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        c
        for c in texto
        if unicodedata.category(c)
        != "Mn"
    )

    texto = re.sub(
        r"[^A-Z0-9]",
        "",
        texto
    )

    return texto


def similar(
    texto,
    objetivo,
    limite=0.70
):

    a = normalizar_texto(
        texto
    )

    b = normalizar_texto(
        objetivo
    )

    if not a or not b:
        return False

    if a == b:
        return True

    ratio = SequenceMatcher(
        None,
        a,
        b
    ).ratio()

    return ratio >= limite


# ============================================================
# EXTRAER DATOS GENERALES
# ============================================================

def extraer_documento(texto):

    texto_upper = texto.upper()

    documento = {
        "folio_ocr": None,
        "obra": None,
        "patente": None
    }

    # --------------------------------------------------------
    # FOLIO
    # --------------------------------------------------------

    candidatos = re.findall(
        r"\b\d{6,8}\b",
        texto_upper
    )

    # En guías Transex el número aparece normalmente
    # dos veces cerca de la parte superior.
    frecuencias = {}

    for numero in candidatos:

        frecuencias[numero] = (
            frecuencias.get(
                numero,
                0
            ) + 1
        )

    if frecuencias:

        ordenados = sorted(
            frecuencias.items(),
            key=lambda x: (
                x[1],
                len(x[0])
            ),
            reverse=True
        )

        documento["folio_ocr"] = (
            ordenados[0][0]
        )

    # --------------------------------------------------------
    # PATENTE
    # --------------------------------------------------------

    patrones_patente = [
        r"PATENTE\s*[:\-]?\s*([A-Z]{2,4}[\-\s]?\d{2,4})",
        r"\b([A-Z]{4}[\-\s]?\d{2})\b"
    ]

    for patron in patrones_patente:

        resultado = re.search(
            patron,
            texto_upper
        )

        if resultado:

            patente = resultado.group(
                1
            )

            patente = patente.replace(
                " ",
                "-"
            )

            documento["patente"] = (
                patente
            )

            break

    # --------------------------------------------------------
    # OBRA
    # --------------------------------------------------------

    resultado_obra = re.search(
        r"OBRA\s*[:\|\*]?\s*([A-ZÁÉÍÓÚÑ0-9 \-]{3,40})",
        texto_upper
    )

    if resultado_obra:

        obra = resultado_obra.group(
            1
        )

        obra = re.split(
            r"[\n\r|]",
            obra
        )[0]

        documento["obra"] = (
            obra.strip()
        )

    return documento


# ============================================================
# DETECCIÓN DE TABLA TRANSEX
# ============================================================

def detectar_y_recortar_tabla(
    imagen,
    palabras
):

    ancho, alto = imagen.size

    # --------------------------------------------------------
    # Buscar encabezados.
    # --------------------------------------------------------

    objetivos = {
        "codigo": "CODIGO",
        "descripcion": "DESCRIPCION",
        "unidad": "UM",
        "cantidad": "CANT",
        "precio": "PRECIO",
        "total": "TOTAL"
    }

    encontrados = {}

    for clave, objetivo in objetivos.items():

        mejores = []

        for palabra in palabras:

            if similar(
                palabra["text"],
                objetivo,
                0.62
            ):

                mejores.append(
                    palabra
                )

        if mejores:

            # Preferir encabezados de zona central,
            # evitando totales del pie.
            mejores = sorted(
                mejores,
                key=lambda p: abs(
                    p["top"]
                    - alto * 0.40
                )
            )

            encontrados[clave] = (
                mejores[0]
            )

    # --------------------------------------------------------
    # Determinar parte superior.
    # --------------------------------------------------------

    if len(encontrados) >= 3:

        tops = [
            p["top"]
            for p in encontrados.values()
        ]

        top_tabla = int(
            statistics.median(
                tops
            )
        )

    else:

        # Fallback específico documento Transex.
        top_tabla = int(
            alto * 0.34
        )

    # Empezamos un poco antes del encabezado.
    top_tabla = max(
        0,
        top_tabla - int(
            alto * 0.015
        )
    )

    # --------------------------------------------------------
    # Buscar final de tabla.
    # --------------------------------------------------------

    candidatos_bottom = []

    for palabra in palabras:

        if (
            palabra["top"]
            <= top_tabla
        ):
            continue

        if (
            similar(
                palabra["text"],
                "ITEM",
                0.70
            )
            or
            similar(
                palabra["text"],
                "SOLICITANTE",
                0.70
            )
        ):

            candidatos_bottom.append(
                palabra["top"]
            )

    if candidatos_bottom:

        bottom_tabla = min(
            candidatos_bottom
        )

    else:

        bottom_tabla = int(
            alto * 0.58
        )

    # Margen.
    bottom_tabla = min(
        alto,
        bottom_tabla
        + int(
            alto * 0.01
        )
    )

    if (
        bottom_tabla
        <= top_tabla
    ):

        return None

    # Recortar horizontal completo.
    tabla = imagen.crop(
        (
            0,
            top_tabla,
            ancho,
            bottom_tabla
        )
    )

    return tabla


# ============================================================
# LIMPIAR LÍNEAS TABLA
# ============================================================

def limpiar_lineas_tabla(
    imagen
):

    """
    Elimina parte de las líneas horizontales y verticales
    de la tabla para facilitar OCR.
    """

    gris = np.array(
        imagen.convert("L")
    )

    binaria = cv2.adaptiveThreshold(
        gris,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    horizontal = binaria.copy()

    vertical = binaria.copy()

    escala_horizontal = max(
        20,
        horizontal.shape[1] // 30
    )

    kernel_horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            escala_horizontal,
            1
        )
    )

    horizontal = cv2.morphologyEx(
        horizontal,
        cv2.MORPH_OPEN,
        kernel_horizontal
    )

    escala_vertical = max(
        20,
        vertical.shape[0] // 8
    )

    kernel_vertical = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            1,
            escala_vertical
        )
    )

    vertical = cv2.morphologyEx(
        vertical,
        cv2.MORPH_OPEN,
        kernel_vertical
    )

    lineas = cv2.bitwise_or(
        horizontal,
        vertical
    )

    limpia = gris.copy()

    limpia[
        lineas > 0
    ] = 255

    pil = Image.fromarray(
        limpia
    )

    pil = ImageOps.autocontrast(
        pil
    )

    ancho, alto = pil.size

    # Agrandar tabla para mejorar números.
    pil = pil.resize(
        (
            int(ancho * 1.5),
            int(alto * 1.5)
        ),
        Image.Resampling.LANCZOS
    )

    return pil


# ============================================================
# INTERPRETAR TABLA TRANSEX
# ============================================================

def interpretar_tabla_transex(
    palabras,
    ancho
):

    if not palabras:
        return []

    # --------------------------------------------------------
    # Determinar altura típica del texto.
    # --------------------------------------------------------

    alturas = [
        p["height"]
        for p in palabras
        if p["height"] > 0
    ]

    altura_media = (
        statistics.median(alturas)
        if alturas
        else 20
    )

    tolerancia_y = max(
        12,
        int(
            altura_media * 0.8
        )
    )

    # --------------------------------------------------------
    # Agrupar palabras por fila.
    # --------------------------------------------------------

    palabras_ordenadas = sorted(
        palabras,
        key=lambda p: (
            p["top"],
            p["left"]
        )
    )

    filas = []

    for palabra in palabras_ordenadas:

        centro_y = (
            palabra["top"]
            + palabra["height"] / 2
        )

        agregada = False

        for fila in filas:

            if abs(
                fila["y"] - centro_y
            ) <= tolerancia_y:

                fila["palabras"].append(
                    palabra
                )

                centros = [
                    p["top"]
                    + p["height"] / 2
                    for p in fila["palabras"]
                ]

                fila["y"] = (
                    sum(centros)
                    / len(centros)
                )

                agregada = True

                break

        if not agregada:

            filas.append({
                "y": centro_y,
                "palabras": [
                    palabra
                ]
            })

    # --------------------------------------------------------
    # Límites de columnas.
    #
    # Valores relativos aproximados según formato Transex.
    # --------------------------------------------------------

    limites = {
        "codigo_fin": 0.17,
        "descripcion_fin": 0.65,
        "unidad_fin": 0.72,
        "cantidad_fin": 0.80,
        "precio_fin": 0.90
    }

    resultados = []

    for fila in filas:

        palabras_fila = sorted(
            fila["palabras"],
            key=lambda p: p["left"]
        )

        texto_fila = " ".join(
            p["text"]
            for p in palabras_fila
        ).strip()

        normalizado = normalizar_texto(
            texto_fila
        )

        # ----------------------------------------------------
        # Ignorar encabezados.
        # ----------------------------------------------------

        if (
            "CODIGO" in normalizado
            or
            "DESCRIPCION" in normalizado
            or
            "PRECIO" in normalizado
        ):
            continue

        # Ignorar fila inferior.
        if (
            normalizado.startswith(
                "ITEM"
            )
            or
            "SOLICITANTE" in normalizado
        ):
            continue

        columnas = {
            "codigo": [],
            "descripcion": [],
            "unidad": [],
            "cantidad": [],
            "precio": [],
            "total": []
        }

        for palabra in palabras_fila:

            centro_x = (
                palabra["left"]
                + palabra["width"] / 2
            )

            proporcion = (
                centro_x / ancho
            )

            if (
                proporcion
                < limites["codigo_fin"]
            ):

                columnas["codigo"].append(
                    palabra["text"]
                )

            elif (
                proporcion
                < limites[
                    "descripcion_fin"
                ]
            ):

                columnas[
                    "descripcion"
                ].append(
                    palabra["text"]
                )

            elif (
                proporcion
                < limites[
                    "unidad_fin"
                ]
            ):

                columnas[
                    "unidad"
                ].append(
                    palabra["text"]
                )

            elif (
                proporcion
                < limites[
                    "cantidad_fin"
                ]
            ):

                columnas[
                    "cantidad"
                ].append(
                    palabra["text"]
                )

            elif (
                proporcion
                < limites[
                    "precio_fin"
                ]
            ):

                columnas[
                    "precio"
                ].append(
                    palabra["text"]
                )

            else:

                columnas[
                    "total"
                ].append(
                    palabra["text"]
                )

        codigo = unir_columna(
            columnas["codigo"]
        )

        descripcion = unir_columna(
            columnas["descripcion"]
        )

        unidad = unir_columna(
            columnas["unidad"]
        )

        cantidad_raw = unir_columna(
            columnas["cantidad"]
        )

        precio_raw = unir_columna(
            columnas["precio"]
        )

        total_raw = unir_columna(
            columnas["total"]
        )

        # ----------------------------------------------------
        # Solo considerar posibles líneas de producto.
        # ----------------------------------------------------

        if (
            not codigo
            and not descripcion
        ):
            continue

        # Transex normalmente usa códigos
        # alfanuméricos cortos.
        parece_producto = (
            bool(
                re.search(
                    r"[A-Z0-9]",
                    codigo.upper()
                )
            )
            and
            len(descripcion) >= 3
        )

        if not parece_producto:
            continue

        cantidad = convertir_decimal_chileno(
            cantidad_raw
        )

        precio = convertir_entero_chileno(
            precio_raw
        )

        total_ocr = convertir_entero_chileno(
            total_raw
        )

        total_calculado = None

        if (
            cantidad is not None
            and precio is not None
        ):

            total_calculado = round(
                cantidad * precio
            )

        # Confianza media de la fila.
        confianzas = [
            p["conf"]
            for p in palabras_fila
            if p["conf"] >= 0
        ]

        confianza = (
            round(
                sum(confianzas)
                / len(confianzas),
                1
            )
            if confianzas
            else None
        )

        resultados.append({

            "codigo":
                limpiar_texto_campo(
                    codigo
                ),

            "descripcion":
                limpiar_texto_campo(
                    descripcion
                ),

            "unidad_ocr":
                limpiar_texto_campo(
                    unidad
                ),

            "cantidad_raw":
                cantidad_raw,

            "cantidad":
                cantidad,

            "precio_raw":
                precio_raw,

            "precio":
                precio,

            "total_raw":
                total_raw,

            "total_ocr":
                total_ocr,

            "total_calculado":
                total_calculado,

            "confianza_ocr":
                confianza,

            "fila_raw":
                texto_fila

        })

    # --------------------------------------------------------
    # Filtrar basura evidente.
    # --------------------------------------------------------

    resultados_limpios = []

    for item in resultados:

        texto = (
            item["codigo"]
            + " "
            + item["descripcion"]
        ).upper()

        if (
            "HORMIGON" in texto
            or
            "CARGA" in texto
            or
            item["cantidad"] is not None
            or
            item["precio"] is not None
        ):

            resultados_limpios.append(
                item
            )

    return resultados_limpios


# ============================================================
# UTILIDADES TABLA
# ============================================================

def unir_columna(valores):

    return " ".join(
        str(v).strip()
        for v in valores
        if str(v).strip()
    ).strip()


def limpiar_texto_campo(texto):

    texto = re.sub(
        r"\s+",
        " ",
        str(texto or "")
    )

    return texto.strip(
        " |[]{}"
    )


def convertir_decimal_chileno(texto):

    if not texto:
        return None

    valor = str(texto)

    valor = valor.replace(
        " ",
        ""
    )

    # Mantener solo dígitos, coma y punto.
    valor = re.sub(
        r"[^0-9,.]",
        "",
        valor
    )

    if not valor:
        return None

    try:

        # Para cantidad esperamos algo como:
        # 3,00
        # 3.00

        if (
            "," in valor
            and "." not in valor
        ):

            valor = valor.replace(
                ",",
                "."
            )

        elif (
            "." in valor
            and "," not in valor
        ):

            # 3.00 => decimal
            partes = valor.split(".")

            if (
                len(partes) == 2
                and len(partes[1]) <= 2
            ):
                pass

            else:
                valor = valor.replace(
                    ".",
                    ""
                )

        elif (
            "." in valor
            and "," in valor
        ):

            # Formato chileno:
            # 1.234,50
            valor = valor.replace(
                ".",
                ""
            ).replace(
                ",",
                "."
            )

        return float(valor)

    except Exception:
        return None


def convertir_entero_chileno(texto):

    if not texto:
        return None

    valor = str(texto)

    valor = re.sub(
        r"[^0-9]",
        "",
        valor
    )

    if not valor:
        return None

    try:
        return int(valor)

    except Exception:
        return None
