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
        "version": "2.1-transex"
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
    # 2. Preparar imagen
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
    # 6. Detectar zona de productos
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

    rgb = np.array(
        imagen.convert("RGB")
    )

    original = rgb.copy()

    alto, ancho = rgb.shape[:2]

    escala = 1200 / max(
        alto,
        ancho
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

        if area < area_imagen * 0.45:
            continue

        pagina = aproximado.reshape(
            4,
            2
        ).astype(np.float32)

        break

    if pagina is None:
        return imagen

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

    suma = puntos.sum(
        axis=1
    )

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

            "text":
                texto,

            "normalizado":
                normalizar_texto(
                    texto
                ),

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
# DATOS GENERALES DOCUMENTO
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

    resultado_folio = re.search(
        r"N[°º]?\s*(\d{6,8})",
        texto_upper
    )

    if resultado_folio:

        documento["folio_ocr"] = (
            resultado_folio.group(1)
        )

    else:

        candidatos = re.findall(
            r"\b\d{6,8}\b",
            texto_upper
        )

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
        r"OBRA\s*[:|*]?\s*([A-ZÁÉÍÓÚÑ0-9 \-]{3,40})",
        texto_upper
    )

    if resultado_obra:

        obra = resultado_obra.group(
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

        documento["obra"] = (
            obra.strip()
        )

    return documento


# ============================================================
# IDENTIFICAR CÓDIGO PRODUCTO TRANSEX
# ============================================================

def es_codigo_producto_transex(
    codigo
):

    codigo = str(
        codigo or ""
    ).strip().upper()

    codigo = codigo.strip(
        "|[]{}();:,."
    )

    # --------------------------------------------------------
    # Código numérico:
    #
    # 4489
    # 12345
    # etc.
    # --------------------------------------------------------

    if re.fullmatch(
        r"\d{3,7}",
        codigo
    ):

        return True

    # --------------------------------------------------------
    # Código alfanumérico compuesto:
    #
    # CAR-INCOMP
    # ABC-123
    # etc.
    # --------------------------------------------------------

    if re.fullmatch(
        r"[A-Z0-9]{2,15}-[A-Z0-9\-]{2,20}",
        codigo
    ):

        return True

    return False


# ============================================================
# DETECTAR Y RECORTAR TABLA
# ============================================================

def detectar_y_recortar_tabla(
    imagen,
    palabras
):

    ancho, alto = imagen.size

    # --------------------------------------------------------
    # Primero buscamos códigos que parezcan productos.
    # --------------------------------------------------------

    candidatos_productos = []

    for palabra in palabras:

        if not es_codigo_producto_transex(
            palabra["text"]
        ):

            continue

        proporcion_y = (
            palabra["top"]
            / alto
        )

        # Los productos de Transex normalmente
        # están en la zona central.
        if (
            0.25
            <= proporcion_y
            <= 0.65
        ):

            candidatos_productos.append(
                palabra
            )

    # --------------------------------------------------------
    # Si detectamos un producto, usamos ese punto.
    # --------------------------------------------------------

    if candidatos_productos:

        primer_producto = min(
            candidatos_productos,
            key=lambda p: p["top"]
        )

        top_tabla = (
            primer_producto["top"]
            - primer_producto["height"] * 2
        )

        top_tabla = int(
            max(
                0,
                top_tabla
            )
        )

    else:

        # ----------------------------------------------------
        # Buscar OBSERVACIONES
        # ----------------------------------------------------

        observaciones = []

        for palabra in palabras:

            if similar(
                palabra["text"],
                "OBSERVACIONES",
                0.65
            ):

                observaciones.append(
                    palabra
                )

        if observaciones:

            obs = min(
                observaciones,
                key=lambda p: p["top"]
            )

            top_tabla = int(
                obs["top"]
                + obs["height"] * 1.2
            )

        else:

            top_tabla = int(
                alto * 0.40
            )

    # --------------------------------------------------------
    # Buscar fin tabla: ITEM / SOLICITANTE
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
                0.72
            )
            or
            similar(
                palabra["text"],
                "SOLICITANTE",
                0.72
            )
        ):

            candidatos_bottom.append(
                palabra["top"]
            )

    if candidatos_bottom:

        bottom_tabla = min(
            candidatos_bottom
        )

        bottom_tabla = int(
            bottom_tabla
            + alto * 0.01
        )

    else:

        bottom_tabla = int(
            alto * 0.58
        )

    if (
        bottom_tabla
        <= top_tabla
    ):

        return None

    tabla = imagen.crop(
        (
            0,
            top_tabla,
            ancho,
            min(
                bottom_tabla,
                alto
            )
        )
    )

    return tabla


# ============================================================
# LIMPIAR LÍNEAS TABLA
# ============================================================

def limpiar_lineas_tabla(
    imagen
):

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

    kernel_horizontal = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                escala_horizontal,
                1
            )
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

    kernel_vertical = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                escala_vertical
            )
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

    alturas = [
        p["height"]
        for p in palabras
        if p["height"] > 0
    ]

    altura_media = (
        statistics.median(
            alturas
        )
        if alturas
        else 20
    )

    tolerancia_y = max(
        12,
        int(
            altura_media * 0.8
        )
    )

    palabras_ordenadas = sorted(
        palabras,
        key=lambda p: (
            p["top"],
            p["left"]
        )
    )

    filas = []

    # --------------------------------------------------------
    # Agrupar palabras por línea horizontal.
    # --------------------------------------------------------

    for palabra in palabras_ordenadas:

        centro_y = (
            palabra["top"]
            + palabra["height"] / 2
        )

        agregada = False

        for fila in filas:

            if abs(
                fila["y"]
                - centro_y
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
    # Posiciones aproximadas columnas Transex.
    # --------------------------------------------------------

    limites = {

        "codigo_fin":
            0.17,

        "descripcion_fin":
            0.65,

        "unidad_fin":
            0.72,

        "cantidad_fin":
            0.80,

        "precio_fin":
            0.90

    }

    resultados = []

    # --------------------------------------------------------
    # Analizar cada fila.
    # --------------------------------------------------------

    for fila in filas:

        palabras_fila = sorted(
            fila["palabras"],
            key=lambda p: p["left"]
        )

        texto_fila = " ".join(
            p["text"]
            for p in palabras_fila
        ).strip()

        if not texto_fila:
            continue

        columnas = {

            "codigo": [],
            "descripcion": [],
            "unidad": [],
            "cantidad": [],
            "precio": [],
            "total": []

        }

        # ----------------------------------------------------
        # Separar según coordenada horizontal.
        # ----------------------------------------------------

        for palabra in palabras_fila:

            centro_x = (
                palabra["left"]
                + palabra["width"] / 2
            )

            proporcion = (
                centro_x
                / ancho
            )

            if (
                proporcion
                < limites["codigo_fin"]
            ):

                columnas[
                    "codigo"
                ].append(
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

        codigo = limpiar_texto_campo(
            unir_columna(
                columnas["codigo"]
            )
        )

        # ----------------------------------------------------
        # FILTRO PRINCIPAL
        #
        # Aquí desaparecen:
        #
        # COMUNA
        # SS
        # A
        #
        # etc.
        # ----------------------------------------------------

        if not es_codigo_producto_transex(
            codigo
        ):

            continue

        descripcion = limpiar_texto_campo(
            unir_columna(
                columnas["descripcion"]
            )
        )

        unidad_ocr = limpiar_texto_campo(
            unir_columna(
                columnas["unidad"]
            )
        )

        cantidad_raw = limpiar_texto_campo(
            unir_columna(
                columnas["cantidad"]
            )
        )

        precio_raw = limpiar_texto_campo(
            unir_columna(
                columnas["precio"]
            )
        )

        total_raw = limpiar_texto_campo(
            unir_columna(
                columnas["total"]
            )
        )

        # ----------------------------------------------------
        # Limpiar descripción específicamente Transex.
        # ----------------------------------------------------

        descripcion = (
            limpiar_descripcion_transex(
                descripcion
            )
        )

        cantidad = convertir_decimal_chileno(
            cantidad_raw
        )

        precio = convertir_entero_chileno(
            precio_raw
        )

        total_ocr = convertir_entero_chileno(
            total_raw
        )

        # ----------------------------------------------------
        # Una línea de producto debe tener
        # cantidad y precio.
        # ----------------------------------------------------

        if (
            cantidad is None
            or precio is None
        ):

            continue

        # ----------------------------------------------------
        # Evitar cantidades absurdas provenientes del OCR.
        #
        # Para hormigón normalmente hablamos de
        # cantidades bastante menores.
        # ----------------------------------------------------

        if (
            cantidad <= 0
            or cantidad > 1000
        ):

            continue

        if precio <= 0:

            continue

        # ----------------------------------------------------
        # Total calculado.
        # ----------------------------------------------------

        total_calculado = round(
            cantidad * precio
        )

        # ----------------------------------------------------
        # Unidad normalizada.
        # ----------------------------------------------------

        unidad = normalizar_unidad_transex(
            unidad_ocr,
            descripcion
        )

        # ----------------------------------------------------
        # Validar total OCR.
        # ----------------------------------------------------

        total_coincide = None

        if total_ocr is not None:

            total_coincide = (
                total_ocr
                == total_calculado
            )

        # ----------------------------------------------------
        # Confianza media Tesseract.
        # ----------------------------------------------------

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
                codigo,

            "descripcion":
                descripcion,

            "unidad":
                unidad,

            "unidad_ocr":
                unidad_ocr,

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

            "total_coincide":
                total_coincide,

            "confianza_ocr":
                confianza,

            "fila_raw":
                texto_fila

        })

    return resultados


# ============================================================
# LIMPIAR DESCRIPCIÓN PRODUCTO
# ============================================================

def limpiar_descripcion_transex(
    descripcion
):

    texto = re.sub(
        r"\s+",
        " ",
        str(
            descripcion or ""
        )
    ).strip()

    texto_upper = texto.upper()

    # --------------------------------------------------------
    # HORMIGÓN
    #
    # Si encontramos algo como:
    #
    # HORMIGON GR20-90%-40 C/08 e
    #
    # cortamos después de C/08.
    # --------------------------------------------------------

    if "HORMIGON" in texto_upper:

        resultado = re.search(
            r"(HORMIGON\s+.*?C/\d{1,3})",
            texto_upper
        )

        if resultado:

            return resultado.group(
                1
            ).strip()

    # --------------------------------------------------------
    # CARGA INCOMPLETA
    # --------------------------------------------------------

    if (
        "CARGA"
        in texto_upper
        and
        "INCOMPLETA"
        in texto_upper
    ):

        return "CARGA INCOMPLETA"

    # --------------------------------------------------------
    # Eliminar ruido de tokens de 1 carácter
    # al final.
    # --------------------------------------------------------

    tokens = texto.split()

    while (
        tokens
        and
        len(tokens[-1]) <= 1
    ):

        tokens.pop()

    return " ".join(
        tokens
    ).strip()


# ============================================================
# NORMALIZAR UNIDAD
# ============================================================

def normalizar_unidad_transex(
    unidad_ocr,
    descripcion
):

    unidad = normalizar_texto(
        unidad_ocr
    )

    descripcion_upper = str(
        descripcion or ""
    ).upper()

    # --------------------------------------------------------
    # Si Tesseract realmente vio M3.
    # --------------------------------------------------------

    if unidad in {
        "M3",
        "M",
        "M³"
    }:

        return "M3"

    # --------------------------------------------------------
    # Para el MVP Transex:
    #
    # Hormigón y Carga Incompleta se expresan
    # en M3.
    #
    # No estamos diciendo que el OCR lo leyó.
    # Estamos normalizando por tipo de ítem.
    # --------------------------------------------------------

    if (
        "HORMIGON"
        in descripcion_upper
    ):

        return "M3"

    if (
        "CARGA INCOMPLETA"
        in descripcion_upper
    ):

        return "M3"

    # --------------------------------------------------------
    # Si no sabemos, conservar OCR.
    # --------------------------------------------------------

    return (
        unidad_ocr
        if unidad_ocr
        else None
    )


# ============================================================
# UTILIDADES
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
        str(
            texto or ""
        )
    )

    return texto.strip(
        " |[]{}"
    )


# ============================================================
# NÚMERO DECIMAL CHILENO
# ============================================================

def convertir_decimal_chileno(
    texto
):

    if not texto:
        return None

    valor = str(
        texto
    )

    valor = valor.replace(
        " ",
        ""
    )

    valor = re.sub(
        r"[^0-9,.]",
        "",
        valor
    )

    if not valor:
        return None

    try:

        if (
            ","
            in valor
            and
            "."
            not in valor
        ):

            valor = valor.replace(
                ",",
                "."
            )

        elif (
            "."
            in valor
            and
            ","
            not in valor
        ):

            partes = valor.split(
                "."
            )

            if (
                len(partes) == 2
                and
                len(partes[1]) <= 2
            ):

                pass

            else:

                valor = valor.replace(
                    ".",
                    ""
                )

        elif (
            "."
            in valor
            and
            ","
            in valor
        ):

            valor = valor.replace(
                ".",
                ""
            ).replace(
                ",",
                "."
            )

        return float(
            valor
        )

    except Exception:

        return None


# ============================================================
# ENTERO CHILENO
# ============================================================

def convertir_entero_chileno(
    texto
):

    if not texto:
        return None

    valor = str(
        texto
    )

    valor = re.sub(
        r"[^0-9]",
        "",
        valor
    )

    if not valor:
        return None

    try:

        return int(
            valor
        )

    except Exception:

        return None
