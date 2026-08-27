import io
import os
import re
import tempfile
import unicodedata
from datetime import datetime

import cv2
import numpy as np
import pytesseract
import pypdfium2 as pdfium

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageOps


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
        "version": "2.5-transex"
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

        if tipo == FORMATO_PDF:

            resultado = procesar_pdf(
                datos
            )

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

    imagen = ImageOps.exif_transpose(
        imagen
    )

    imagen_documento = corregir_perspectiva(
        imagen
    )

    imagen_preparada = preparar_imagen(
        imagen_documento
    )

    # --------------------------------------------------------
    # OCR COMPLETO
    # --------------------------------------------------------

    texto_completo = ejecutar_ocr_texto(
        imagen_preparada,
        psm=6
    )

    # --------------------------------------------------------
    # OCR TABLA
    # --------------------------------------------------------

    tabla = recortar_tabla_transex(
        imagen_preparada
    )

    tabla_texto = ""

    if tabla is not None:

        tabla = preparar_recorte(
            tabla
        )

        tabla_texto = ejecutar_ocr_texto(
            tabla,
            psm=6
        )

    # --------------------------------------------------------
    # COMBINAMOS AMBAS LECTURAS
    # --------------------------------------------------------

    texto_combinado = (
        texto_completo
        + "\n"
        + tabla_texto
    )

    # --------------------------------------------------------
    # DOCUMENTO
    # --------------------------------------------------------

    documento = extraer_documento(
        texto_combinado
    )

    # --------------------------------------------------------
    # PRODUCTOS
    # --------------------------------------------------------

    detalle = interpretar_productos_transex(
        texto_completo,
        tabla_texto
    )

    return {
        "text": texto_completo,
        "documento": documento,
        "detalle": detalle,
        "tabla_texto": tabla_texto
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
                int(ancho * factor),
                int(alto * factor)
            ),
            Image.Resampling.LANCZOS
        )

    return imagen


def preparar_recorte(
    imagen: Image.Image
):

    imagen = imagen.convert(
        "L"
    )

    imagen = ImageOps.autocontrast(
        imagen
    )

    ancho, alto = imagen.size

    imagen = imagen.resize(
        (
            int(ancho * 1.5),
            int(alto * 1.5)
        ),
        Image.Resampling.LANCZOS
    )

    return imagen


# ============================================================
# OCR
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


# ============================================================
# PERSPECTIVA
# ============================================================

def corregir_perspectiva(
    imagen: Image.Image
):

    rgb = np.array(
        imagen.convert("RGB")
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

    alto_izquierdo = np.linalg.norm(
        bl - tl
    )

    alto_derecho = np.linalg.norm(
        br - tr
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
# DOCUMENTO
# ============================================================

def extraer_documento(
    texto
):

    return {
        "folio_ocr":
            extraer_folio(
                texto
            ),

        "folio_parcial_ocr":
            extraer_folio_parcial(
                texto
            ),

        "fecha_emision_ocr":
            extraer_fecha_emision(
                texto
            ),

        "rut_emisor_ocr":
            extraer_rut_emisor(
                texto
            ),

        "total_documento_ocr":
            extraer_total_documento(
                texto
            ),

        "obra":
            extraer_obra(
                texto
            ),

        "patente":
            extraer_patente(
                texto
            )
    }


# ============================================================
# FOLIO
# ============================================================

def extraer_folio(
    texto
):

    texto_upper = texto.upper()

    candidatos = re.findall(
        r"N\s*[°º]?\s*(\d{6})\b",
        texto_upper
    )

    if not candidatos:

        return None

    frecuencias = {}

    for numero in candidatos:

        frecuencias[numero] = (
            frecuencias.get(
                numero,
                0
            ) + 1
        )

    return max(
        frecuencias,
        key=frecuencias.get
    )


def extraer_folio_parcial(
    texto
):

    # --------------------------------------------------------
    # Para casos como:
    #
    # N 69971
    #
    # donde OCR perdió un dígito.
    # --------------------------------------------------------

    resultado = re.search(
        r"\bN\s*[°º]?\s*(\d{5})\b",
        texto.upper()
    )

    if resultado:

        return resultado.group(
            1
        )

    return None


# ============================================================
# FECHA
# ============================================================

def extraer_fecha_emision(
    texto
):

    # --------------------------------------------------------
    # PRIMERO:
    # FECHA EMISION xx/xx/xxxx
    # --------------------------------------------------------

    resultado = re.search(
        r"FECHA\s+EMISION"
        r"\s*[:\-]?\s*"
        r"(\d{2}/\d{2}/\d{4})",
        texto.upper()
    )

    if resultado:

        fecha = validar_fecha(
            resultado.group(1)
        )

        if fecha:

            return fecha

    # --------------------------------------------------------
    # FALLBACK:
    #
    # buscamos todas las fechas válidas
    # y usamos la más antigua.
    #
    # Normalmente:
    #
    # emisión = agosto
    # vencimiento = septiembre
    # --------------------------------------------------------

    candidatos = re.findall(
        r"\b\d{2}/\d{2}/\d{4}\b",
        texto
    )

    fechas = []

    for candidato in candidatos:

        fecha = validar_fecha(
            candidato
        )

        if fecha:

            fechas.append(
                fecha
            )

    if not fechas:

        return None

    fechas_objeto = []

    for fecha in fechas:

        try:

            objeto = datetime.strptime(
                fecha,
                "%d/%m/%Y"
            )

            fechas_objeto.append(
                (
                    objeto,
                    fecha
                )
            )

        except Exception:

            pass

    if not fechas_objeto:

        return None

    fechas_objeto.sort(
        key=lambda x: x[0]
    )

    return fechas_objeto[0][1]


def validar_fecha(
    texto
):

    try:

        fecha = datetime.strptime(
            texto,
            "%d/%m/%Y"
        )

        return fecha.strftime(
            "%d/%m/%Y"
        )

    except Exception:

        return None


# ============================================================
# RUT EMISOR
# ============================================================

def extraer_rut_emisor(
    texto
):

    texto_upper = texto.upper()

    # --------------------------------------------------------
    # TRANSEX
    # --------------------------------------------------------

    if re.search(
        r"88[\.\s]*147[\.\s]*"
        r"[56][0O][0O][\-\s]*2",
        texto_upper
    ):

        return "88147600-2"

    candidatos = re.findall(
        r"\b\d{1,2}"
        r"[\.\s]?\d{3}"
        r"[\.\s]?\d{3}"
        r"[\-\s]?[0-9K]\b",
        texto_upper
    )

    if candidatos:

        return normalizar_rut(
            candidatos[0]
        )

    return None


def normalizar_rut(
    rut
):

    rut = str(
        rut or ""
    ).upper()

    rut = re.sub(
        r"[^0-9K]",
        "",
        rut
    )

    if len(rut) < 2:

        return None

    return (
        rut[:-1]
        + "-"
        + rut[-1]
    )


# ============================================================
# TOTAL DOCUMENTO
# ============================================================

def extraer_total_documento(
    texto
):

    candidatos = []

    for linea in texto.splitlines():

        linea_upper = linea.upper()

        if "SUBTOTAL" in linea_upper:
            continue

        if not re.search(
            r"\bTOTAL\b",
            linea_upper
        ):
            continue

        montos = extraer_montos(
            linea
        )

        for monto in montos:

            if monto >= 1000:

                candidatos.append(
                    monto
                )

    if candidatos:

        return max(
            candidatos
        )

    return None


# ============================================================
# OBRA
# ============================================================

def extraer_obra(
    texto
):

    texto_upper = texto.upper()

    if "HOTEL BELLET" in texto_upper:

        return "HOTEL BELLET"

    resultado = re.search(
        r"OBRA\s*[:|*]?\s*"
        r"([A-Z0-9 \-]{3,40})",
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
        ).strip()

        if len(obra) >= 3:

            return obra

    return None


# ============================================================
# PATENTE
# ============================================================

def extraer_patente(
    texto
):

    texto_upper = texto.upper()

    # --------------------------------------------------------
    # CERCA DE PALABRA PATENTE
    # --------------------------------------------------------

    resultado = re.search(
        r"PATENTE"
        r"[\s:>\-|]*"
        r"([A-Z]{2,4}"
        r"[\-\s]?"
        r"\d{2,4})",
        texto_upper
    )

    if resultado:

        return normalizar_patente(
            resultado.group(1)
        )

    # --------------------------------------------------------
    # FALLBACK GENERAL
    #
    # PKYS-62
    # XE-9423
    # BKK-52
    # --------------------------------------------------------

    candidatos = re.findall(
        r"\b("
        r"[A-Z]{2,4}"
        r"-"
        r"\d{2,4}"
        r")\b",
        texto_upper
    )

    for candidato in candidatos:

        # Evitar código hormigón
        if candidato.startswith(
            "GR"
        ):
            continue

        return normalizar_patente(
            candidato
        )

    return None


def normalizar_patente(
    patente
):

    patente = str(
        patente or ""
    ).upper().strip()

    patente = patente.replace(
        " ",
        ""
    )

    # DLBT70
    resultado = re.fullmatch(
        r"([A-Z]{4})(\d{2})",
        patente
    )

    if resultado:

        return (
            resultado.group(1)
            + "-"
            + resultado.group(2)
        )

    # XE9423
    resultado = re.fullmatch(
        r"([A-Z]{2})(\d{4})",
        patente
    )

    if resultado:

        return (
            resultado.group(1)
            + "-"
            + resultado.group(2)
        )

    return patente


# ============================================================
# REGLA CANTIDAD
# ============================================================

def normalizar_medio_m3(
    cantidad
):

    if cantidad is None:

        return None

    try:

        cantidad = float(
            cantidad
        )

    except Exception:

        return None

    if cantidad <= 0:

        return None

    return round(
        cantidad * 2
    ) / 2


def cantidad_es_multiplo_medio(
    cantidad
):

    if cantidad is None:

        return False

    normalizada = normalizar_medio_m3(
        cantidad
    )

    if normalizada is None:

        return False

    return abs(
        float(cantidad)
        - normalizada
    ) < 0.001


# ============================================================
# PARSER PRODUCTOS
# ============================================================

def interpretar_productos_transex(
    texto_completo,
    tabla_texto
):

    candidatos = []

    candidatos.extend(
        buscar_productos_en_texto(
            texto_completo,
            "TEXTO_OCR"
        )
    )

    candidatos.extend(
        buscar_productos_en_texto(
            tabla_texto,
            "TABLA_OCR"
        )
    )

    agrupados = {}

    for candidato in candidatos:

        codigo = candidato[
            "codigo"
        ]

        if codigo not in agrupados:

            agrupados[
                codigo
            ] = []

        agrupados[
            codigo
        ].append(
            candidato
        )

    resultados = []

    for codigo, opciones in agrupados.items():

        resultado = resolver_producto(
            codigo,
            opciones
        )

        if resultado:

            resultados.append(
                resultado
            )

    return resultados


# ============================================================
# GENERAR BLOQUES ALREDEDOR DE PRODUCTO
# ============================================================

def generar_bloques(
    texto
):

    lineas = [
        re.sub(
            r"\s+",
            " ",
            linea
        ).strip()
        for linea in texto.splitlines()
    ]

    bloques = []

    for indice, linea in enumerate(
        lineas
    ):

        if not linea:
            continue

        linea_upper = linea.upper()

        # ----------------------------------------------------
        # HORMIGÓN
        # ----------------------------------------------------

        if re.search(
            r"\b4489\b",
            linea_upper
        ):

            partes = [
                linea
            ]

            # Agregamos hasta dos líneas siguientes.
            for offset in [1, 2]:

                posicion = (
                    indice + offset
                )

                if posicion < len(lineas):

                    siguiente = lineas[
                        posicion
                    ]

                    if siguiente:

                        partes.append(
                            siguiente
                        )

            bloques.append({
                "tipo": "HORMIGON",
                "texto": " ".join(
                    partes
                )
            })

        # ----------------------------------------------------
        # CARGA INCOMPLETA
        # ----------------------------------------------------

        if (
            "CARGA"
            in linea_upper
            and
            (
                "INCOMP"
                in linea_upper
                or
                "INCO"
                in linea_upper
            )
        ):

            partes = [
                linea
            ]

            if (
                indice + 1
                < len(lineas)
            ):

                siguiente = lineas[
                    indice + 1
                ]

                if siguiente:

                    partes.append(
                        siguiente
                    )

            bloques.append({
                "tipo": "CARGA",
                "texto": " ".join(
                    partes
                )
            })

    return bloques


# ============================================================
# BUSCAR PRODUCTOS
# ============================================================

def buscar_productos_en_texto(
    texto,
    fuente
):

    resultados = []

    if not texto:

        return resultados

    bloques = generar_bloques(
        texto
    )

    for bloque in bloques:

        if bloque["tipo"] == "HORMIGON":

            item = interpretar_bloque_hormigon(
                bloque["texto"],
                fuente
            )

        else:

            item = interpretar_bloque_carga(
                bloque["texto"],
                fuente
            )

        if item:

            resultados.append(
                item
            )

    return resultados


# ============================================================
# HORMIGÓN
# ============================================================

def interpretar_bloque_hormigon(
    texto,
    fuente
):

    texto_upper = texto.upper()

    if not re.search(
        r"\b4489\b",
        texto_upper
    ):

        return None

    descripcion = extraer_descripcion_hormigon(
        texto_upper
    )

    # --------------------------------------------------------
    # EVITAR NÚMEROS DEL CÓDIGO DEL HORMIGÓN
    #
    # Tomamos texto después de C/08 si existe.
    # --------------------------------------------------------

    resultado = re.search(
        r"C/\d{1,3}",
        texto_upper
    )

    if resultado:

        cola = texto[
            resultado.end():
        ]

    else:

        resultado_codigo = re.search(
            r"\b4489\b",
            texto_upper
        )

        cola = texto[
            resultado_codigo.end():
        ]

    componentes = extraer_componentes_producto(
        cola
    )

    return {
        "codigo":
            "4489",

        "descripcion":
            descripcion,

        "unidad":
            "M3",

        "cantidad_ocr":
            componentes[
                "cantidad_ocr"
            ],

        "montos":
            componentes[
                "montos"
            ],

        "fuente":
            fuente,

        "fila_raw":
            texto
    }


# ============================================================
# CARGA
# ============================================================

def interpretar_bloque_carga(
    texto,
    fuente
):

    texto_upper = texto.upper()

    resultado = re.search(
        r"CARGA\s+INCO[A-Z]*",
        texto_upper
    )

    if resultado:

        cola = texto[
            resultado.end():
        ]

    else:

        cola = texto

    componentes = extraer_componentes_producto(
        cola
    )

    return {
        "codigo":
            "CAR-INCOMP",

        "descripcion":
            "CARGA INCOMPLETA",

        "unidad":
            "M3",

        "cantidad_ocr":
            componentes[
                "cantidad_ocr"
            ],

        "montos":
            componentes[
                "montos"
            ],

        "fuente":
            fuente,

        "fila_raw":
            texto
    }


# ============================================================
# EXTRAER CANTIDAD Y MONTOS
# ============================================================

def extraer_componentes_producto(
    texto
):

    # --------------------------------------------------------
    # CANTIDAD
    #
    # SOLO:
    #
    # x,00
    # x,50
    # x.00
    # x.50
    # --------------------------------------------------------

    cantidades_raw = re.findall(
        r"(?<!\d)"
        r"(\d{1,3}[,.](?:00|50))"
        r"(?!\d)",
        texto
    )

    cantidades = []

    for raw in cantidades_raw:

        cantidad = convertir_decimal(
            raw
        )

        if (
            cantidad is not None
            and
            0 < cantidad <= 100
            and
            cantidad_es_multiplo_medio(
                cantidad
            )
        ):

            cantidades.append(
                cantidad
            )

    cantidades = lista_unica(
        cantidades
    )

    # --------------------------------------------------------
    # MONTOS
    #
    # IMPORTANTE:
    #
    # 83.772 586.404
    #
    # DEBE DAR:
    #
    # 83772
    # 586404
    #
    # y NO:
    #
    # 83772586404
    # --------------------------------------------------------

    montos = extraer_montos(
        texto
    )

    return {
        "cantidad_ocr":
            (
                cantidades[0]
                if cantidades
                else None
            ),

        "montos":
            montos
    }


# ============================================================
# EXTRAER MONTOS CORRECTAMENTE
# ============================================================

def extraer_montos(
    texto
):

    """
    Reconoce por separado:

    83.772
    586.404
    83,772
    251 334
    83778
    670224

    Nunca une:

    83.772 586.404
    """

    patron = re.compile(
        r"(?<![\d.,])"
        r"("
        r"\d{1,3}(?:\.\d{3})+"
        r"|"
        r"\d{1,3}(?:,\d{3})+"
        r"|"
        r"\d{1,3}\s+\d{3}"
        r"|"
        r"\d{4,7}"
        r")"
        r"(?![\d.,])"
    )

    resultados = []

    for coincidencia in patron.finditer(
        texto
    ):

        raw = coincidencia.group(
            1
        )

        valor = convertir_monto(
            raw
        )

        if (
            valor is not None
            and
            1000 <= valor <= 9999999
        ):

            resultados.append(
                valor
            )

    return resultados


# ============================================================
# RESOLVER PRODUCTO
# ============================================================

def resolver_producto(
    codigo,
    opciones
):

    cantidades_ocr = []

    todos_montos = []

    fuentes = []

    filas = []

    descripciones = []

    for opcion in opciones:

        cantidad = opcion.get(
            "cantidad_ocr"
        )

        if cantidad is not None:

            cantidades_ocr.append(
                cantidad
            )

        todos_montos.extend(
            opcion.get(
                "montos",
                []
            )
        )

        fuente = opcion.get(
            "fuente"
        )

        if fuente:

            fuentes.append(
                fuente
            )

        fila = opcion.get(
            "fila_raw"
        )

        if fila:

            filas.append(
                fila
            )

        descripcion = opcion.get(
            "descripcion"
        )

        if descripcion:

            descripciones.append(
                descripcion
            )

    cantidades_ocr = lista_unica(
        cantidades_ocr
    )

    # --------------------------------------------------------
    # Conservamos repetición para saber qué monto
    # apareció más veces.
    # --------------------------------------------------------

    montos_unicos = lista_unica(
        todos_montos
    )

    # --------------------------------------------------------
    # BUSCAR SOLUCIÓN EXACTA:
    #
    # TOTAL / PRECIO =
    #
    # 0,50
    # 1,00
    # 1,50
    # ...
    # --------------------------------------------------------

    solucion = buscar_solucion_matematica(
        montos_unicos,
        cantidades_ocr
    )

    if solucion:

        cantidad = solucion[
            "cantidad"
        ]

        precio = solucion[
            "precio"
        ]

        total_ocr = solucion[
            "total"
        ]

        cantidad_calculada = solucion[
            "cantidad_calculada"
        ]

        cantidad_validada = True

        cantidad_fuente = (
            "CALCULADA_PRECIO_TOTAL"
        )

        coincide_con_ocr = any(
            abs(
                valor - cantidad
            ) < 0.001
            for valor in cantidades_ocr
        )

        cantidad_forzada = (
            not coincide_con_ocr
        )

    else:

        # ----------------------------------------------------
        # FALLBACK:
        #
        # cantidad OCR válida + precio más probable
        # ----------------------------------------------------

        cantidad = (
            cantidades_ocr[0]
            if cantidades_ocr
            else None
        )

        precio = seleccionar_precio(
            todos_montos
        )

        total_ocr = None

        cantidad_calculada = None

        cantidad_validada = False

        cantidad_fuente = (
            "OCR"
            if cantidad is not None
            else None
        )

        cantidad_forzada = False

    # --------------------------------------------------------
    # TOTAL CALCULADO
    # --------------------------------------------------------

    total_calculado = None

    if (
        cantidad is not None
        and
        precio is not None
    ):

        total_calculado = round(
            cantidad
            * precio
        )

    total_coincide = None

    if (
        total_ocr is not None
        and
        total_calculado is not None
    ):

        total_coincide = (
            abs(
                total_calculado
                - total_ocr
            )
            <= 1
        )

    descripcion = seleccionar_descripcion(
        codigo,
        descripciones
    )

    return {
        "codigo":
            codigo,

        "descripcion":
            descripcion,

        "unidad":
            "M3",

        "cantidad_ocr":
            (
                cantidades_ocr[0]
                if cantidades_ocr
                else None
            ),

        "cantidad_calculada":
            cantidad_calculada,

        "cantidad":
            cantidad,

        "cantidad_fuente":
            cantidad_fuente,

        "cantidad_forzada":
            cantidad_forzada,

        "cantidad_validada":
            cantidad_validada,

        "regla_cantidad":
            "MULTIPLO_0_50",

        "precio":
            precio,

        "total_ocr":
            total_ocr,

        "total_calculado":
            total_calculado,

        "total_coincide":
            total_coincide,

        "parser_ok":
            (
                cantidad is not None
                and
                precio is not None
            ),

        "fuente":
            "+".join(
                sorted(
                    set(
                        fuentes
                    )
                )
            ),

        "montos_detectados":
            montos_unicos,

        "fila_raw":
            " || ".join(
                filas
            )
    }


# ============================================================
# BUSCAR SOLUCIÓN MATEMÁTICA
# ============================================================

def buscar_solucion_matematica(
    montos,
    cantidades_ocr
):

    soluciones = []

    # --------------------------------------------------------
    # PROBAMOS TODAS LAS COMBINACIONES.
    #
    # No suponemos que el primer número
    # necesariamente sea el precio.
    # --------------------------------------------------------

    for precio in montos:

        # Precio unitario razonable.
        if not (
            10000
            <= precio
            <= 500000
        ):

            continue

        for total in montos:

            if total <= precio:

                continue

            cantidad_exacta = (
                total
                / precio
            )

            cantidad_normalizada = (
                normalizar_medio_m3(
                    cantidad_exacta
                )
            )

            if cantidad_normalizada is None:

                continue

            if not (
                0.5
                <= cantidad_normalizada
                <= 100
            ):

                continue

            total_validacion = round(
                cantidad_normalizada
                * precio
            )

            diferencia = abs(
                total_validacion
                - total
            )

            # ------------------------------------------------
            # Debe cuadrar matemáticamente.
            # ------------------------------------------------

            if diferencia > 1:

                continue

            puntaje = 1000

            # ------------------------------------------------
            # Si OCR también leyó la misma cantidad,
            # esta combinación gana prioridad.
            # ------------------------------------------------

            for cantidad_ocr in cantidades_ocr:

                if abs(
                    cantidad_ocr
                    - cantidad_normalizada
                ) < 0.001:

                    puntaje += 500

            # ------------------------------------------------
            # Precios más probables de hormigón.
            # ------------------------------------------------

            if (
                30000
                <= precio
                <= 200000
            ):

                puntaje += 50

            soluciones.append({
                "cantidad":
                    cantidad_normalizada,

                "cantidad_calculada":
                    round(
                        cantidad_exacta,
                        6
                    ),

                "precio":
                    precio,

                "total":
                    total,

                "diferencia":
                    diferencia,

                "puntaje":
                    puntaje
            })

    if not soluciones:

        return None

    soluciones.sort(
        key=lambda x: (
            x["puntaje"],
            -x["diferencia"]
        ),
        reverse=True
    )

    return soluciones[0]


# ============================================================
# PRECIO FALLBACK
# ============================================================

def seleccionar_precio(
    montos
):

    if not montos:

        return None

    candidatos = [
        monto
        for monto in montos
        if (
            10000
            <= monto
            <= 200000
        )
    ]

    if not candidatos:

        return None

    # --------------------------------------------------------
    # Elegir el que apareció más veces
    # entre TEXTO_OCR + TABLA_OCR.
    # --------------------------------------------------------

    return max(
        set(candidatos),
        key=candidatos.count
    )


# ============================================================
# DESCRIPCIÓN
# ============================================================

def extraer_descripcion_hormigon(
    texto
):

    resultado = re.search(
        r"(HORMIGON\s+.*?C/\d{1,3})",
        texto
    )

    if resultado:

        return limpiar_texto(
            resultado.group(1)
        )

    resultado = re.search(
        r"(GR\d{1,3}.*?C/\d{1,3})",
        texto
    )

    if resultado:

        return limpiar_texto(
            "HORMIGON "
            + resultado.group(1)
        )

    return "HORMIGON"


def seleccionar_descripcion(
    codigo,
    descripciones
):

    if codigo == "CAR-INCOMP":

        return "CARGA INCOMPLETA"

    completas = [
        descripcion
        for descripcion in descripciones
        if (
            descripcion
            and
            "HORMIGON"
            in descripcion.upper()
            and
            "C/"
            in descripcion.upper()
        )
    ]

    if completas:

        return max(
            completas,
            key=len
        )

    return "HORMIGON"


# ============================================================
# RECORTE TABLA
# ============================================================

def recortar_tabla_transex(
    imagen
):

    ancho, alto = imagen.size

    top = int(
        alto * 0.36
    )

    bottom = int(
        alto * 0.58
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
# CONVERSIONES
# ============================================================

def convertir_decimal(
    texto
):

    if texto is None:

        return None

    valor = str(
        texto
    ).replace(
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
        str(texto)
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
# UTILIDADES
# ============================================================

def lista_unica(
    valores
):

    resultado = []

    for valor in valores:

        if valor not in resultado:

            resultado.append(
                valor
            )

    return resultado


def limpiar_texto(
    texto
):

    return re.sub(
        r"\s+",
        " ",
        str(
            texto or ""
        )
    ).strip()


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
