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
        "version": "2.4-transex"
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

    # --------------------------------------------------------
    # CORREGIR PERSPECTIVA
    # --------------------------------------------------------

    imagen_documento = corregir_perspectiva(
        imagen
    )

    # --------------------------------------------------------
    # PREPARAR IMAGEN
    # --------------------------------------------------------

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
    # OCR ZONA TABLA
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
    # DATOS GENERALES
    # --------------------------------------------------------

    documento = extraer_documento(
        texto_completo
    )

    # --------------------------------------------------------
    # DETALLE
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
# PREPARACIÓN IMAGEN
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
# CORREGIR PERSPECTIVA
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

    # --------------------------------------------------------
    # SOLO BUSCAMOS NÚMERO CERCA DE N / N°
    #
    # No usamos cualquier número de seis dígitos
    # porque podría ser subtotal, sello, etc.
    # --------------------------------------------------------

    patrones = [
        r"N[°º]\s*(\d{6})",
        r"\bN\s*[°º]?\s*(\d{6})\b"
    ]

    candidatos = []

    for patron in patrones:

        candidatos.extend(
            re.findall(
                patron,
                texto_upper
            )
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


# ============================================================
# FECHA EMISIÓN
# ============================================================

def extraer_fecha_emision(
    texto
):

    resultado = re.search(
        r"FECHA\s+EMISION"
        r"\s*[:\-]?\s*"
        r"(\d{2}/\d{2}/\d{4})",
        texto.upper()
    )

    if resultado:

        return resultado.group(
            1
        )

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

    patron_transex = re.search(
        r"88"
        r"[\.\s]*147"
        r"[\.\s]*"
        r"[56][0O][0O]"
        r"[\-\s]*2",
        texto_upper
    )

    if patron_transex:

        return "88147600-2"

    # --------------------------------------------------------
    # GENÉRICO
    # --------------------------------------------------------

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

        linea_upper = (
            linea.upper()
        )

        if "SUBTOTAL" in linea_upper:

            continue

        if not re.search(
            r"\bTOTAL\b",
            linea_upper
        ):

            continue

        montos = re.findall(
            r"\b\d{1,3}"
            r"(?:[.,]\d{3})+\b",
            linea
        )

        for monto in montos:

            valor = convertir_monto(
                monto
            )

            if valor is not None:

                candidatos.append(
                    valor
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

    # --------------------------------------------------------
    # ACTUAL MVP
    # --------------------------------------------------------

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

    return None


def normalizar_patente(
    patente
):

    patente = str(
        patente or ""
    ).upper().strip()

    patente = re.sub(
        r"\s+",
        "",
        patente
    )

    # --------------------------------------------------------
    # NUEVO FORMATO:
    #
    # DLBT70 -> DLBT-70
    # PKYS62 -> PKYS-62
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FORMATO ANTIGUO:
    #
    # XE9423 -> XE-9423
    # --------------------------------------------------------

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

    patente = re.sub(
        r"-+",
        "-",
        patente
    )

    return patente


# ============================================================
# REGLA MATEMÁTICA DE CANTIDAD
# ============================================================

def normalizar_medio_m3(
    cantidad
):

    """
    REGLA DE NEGOCIO:

    La cantidad siempre debe ser:

    0,50
    1,00
    1,50
    2,00
    2,50
    etc.

    Nunca:
    3,20
    6,80
    7,25
    """

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

    try:

        cantidad = float(
            cantidad
        )

    except Exception:

        return False

    cantidad_normalizada = (
        normalizar_medio_m3(
            cantidad
        )
    )

    return abs(
        cantidad
        - cantidad_normalizada
    ) < 0.01


# ============================================================
# PRODUCTOS TRANSEX
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
# BUSCAR LÍNEAS PRODUCTO
# ============================================================

def buscar_productos_en_texto(
    texto,
    fuente
):

    resultados = []

    if not texto:

        return resultados

    lineas = texto.splitlines()

    for linea in lineas:

        linea = re.sub(
            r"\s+",
            " ",
            linea
        ).strip()

        if not linea:

            continue

        linea_upper = linea.upper()

        # ----------------------------------------------------
        # HORMIGÓN CÓDIGO 4489
        # ----------------------------------------------------

        if re.search(
            r"\b4489\b",
            linea_upper
        ):

            item = interpretar_linea_hormigon(
                linea,
                fuente
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
            (
                "INCOMPLETA"
                in linea_upper
                or
                "INCO"
                in linea_upper
            )
        ):

            item = interpretar_linea_carga(
                linea,
                fuente
            )

            if item:

                resultados.append(
                    item
                )

    return resultados


# ============================================================
# INTERPRETAR HORMIGÓN
# ============================================================

def interpretar_linea_hormigon(
    linea,
    fuente
):

    linea_upper = linea.upper()

    if not re.search(
        r"\b4489\b",
        linea_upper
    ):

        return None

    descripcion = extraer_descripcion_hormigon(
        linea_upper
    )

    # --------------------------------------------------------
    # TRABAJAR CON LA PARTE POSTERIOR A C/XX
    #
    # Evita confundir:
    #
    # GR20
    # 90
    # 40
    # 08
    #
    # con cantidades o precios.
    # --------------------------------------------------------

    resultado_fin = re.search(
        r"C/\d{1,3}",
        linea_upper
    )

    if resultado_fin:

        cola = linea[
            resultado_fin.end():
        ]

    else:

        resultado_codigo = re.search(
            r"\b4489\b",
            linea_upper
        )

        cola = linea[
            resultado_codigo.end():
        ]

    componentes = extraer_componentes_producto(
        cola
    )

    return {
        "codigo": "4489",
        "descripcion": descripcion,
        "unidad": "M3",
        "fuente": fuente,
        "cantidad_ocr":
            componentes["cantidad_ocr"],
        "precio_candidatos":
            componentes["montos"][:1],
        "total_candidatos":
            componentes["montos"][1:],
        "montos_candidatos":
            componentes["montos"],
        "fila_raw": linea
    }


# ============================================================
# CARGA INCOMPLETA
# ============================================================

def interpretar_linea_carga(
    linea,
    fuente
):

    linea_upper = linea.upper()

    resultado = re.search(
        r"CARGA\s+INCO[A-Z]*",
        linea_upper
    )

    if resultado:

        cola = linea[
            resultado.end():
        ]

    else:

        cola = linea

    componentes = extraer_componentes_producto(
        cola
    )

    return {
        "codigo": "CAR-INCOMP",
        "descripcion": "CARGA INCOMPLETA",
        "unidad": "M3",
        "fuente": fuente,
        "cantidad_ocr":
            componentes["cantidad_ocr"],
        "precio_candidatos":
            componentes["montos"][:1],
        "total_candidatos":
            componentes["montos"][1:],
        "montos_candidatos":
            componentes["montos"],
        "fila_raw": linea
    }


# ============================================================
# COMPONENTES NUMÉRICOS DE UNA LÍNEA
# ============================================================

def extraer_componentes_producto(
    texto
):

    cantidades = []

    # --------------------------------------------------------
    # CANTIDADES PERFECTAMENTE VÁLIDAS
    #
    # Solo:
    #
    # X,00
    # X,50
    # X.00
    # X.50
    # --------------------------------------------------------

    cantidades_raw = re.findall(
        r"(?<!\d)"
        r"(\d{1,3}[,.](?:00|50))"
        r"(?!\d)",
        texto
    )

    for cantidad_raw in cantidades_raw:

        cantidad = convertir_decimal(
            cantidad_raw
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

    # --------------------------------------------------------
    # CASO:
    #
    # M3 8
    #
    # Por si OCR perdió ",00".
    # --------------------------------------------------------

    resultado_entero = re.search(
        r"M3"
        r"[\s|:\-]*"
        r"(\d{1,2})\b",
        texto.upper()
    )

    if resultado_entero:

        cantidad = float(
            resultado_entero.group(1)
        )

        if (
            0 < cantidad <= 100
        ):

            cantidades.append(
                cantidad
            )

    # Quitar repetidos.
    cantidades = lista_unica(
        cantidades
    )

    # --------------------------------------------------------
    # MONTOS
    #
    # Reconoce:
    #
    # 83.778
    # 586.404
    # 251 334
    # 83778
    # 670224
    # --------------------------------------------------------

    montos_raw = re.findall(
        r"(?<!\d)"
        r"("
        r"\d{1,3}(?:[.\s]\d{3})+"
        r"|"
        r"\d{4,7}"
        r")"
        r"(?!\d)",
        texto
    )

    montos = []

    for monto_raw in montos_raw:

        monto = convertir_monto(
            monto_raw
        )

        if (
            monto is not None
            and
            monto >= 1000
        ):

            montos.append(
                monto
            )

    montos = lista_unica(
        montos
    )

    cantidad_ocr = (
        cantidades[0]
        if cantidades
        else None
    )

    return {
        "cantidad_ocr":
            cantidad_ocr,

        "cantidades_ocr":
            cantidades,

        "montos":
            montos
    }


# ============================================================
# RESOLVER PRODUCTO
# ============================================================

def resolver_producto(
    codigo,
    opciones
):

    if not opciones:

        return None

    # --------------------------------------------------------
    # CANTIDADES LEÍDAS POR OCR
    # --------------------------------------------------------

    cantidades_ocr = []

    precios = []

    totales = []

    fuentes = []

    filas = []

    descripciones = []

    for opcion in opciones:

        fuentes.append(
            opcion.get(
                "fuente"
            )
        )

        filas.append(
            opcion.get(
                "fila_raw"
            )
        )

        descripciones.append(
            opcion.get(
                "descripcion"
            )
        )

        cantidad = opcion.get(
            "cantidad_ocr"
        )

        if cantidad is not None:

            cantidades_ocr.append(
                cantidad
            )

        for precio in opcion.get(
            "precio_candidatos",
            []
        ):

            if (
                precio is not None
                and
                precio >= 1000
            ):

                precios.append(
                    precio
                )

        for total in opcion.get(
            "total_candidatos",
            []
        ):

            if (
                total is not None
                and
                total >= 1000
            ):

                totales.append(
                    total
                )

    cantidades_ocr = lista_unica(
        cantidades_ocr
    )

    precios = lista_unica(
        precios
    )

    totales = lista_unica(
        totales
    )

    # --------------------------------------------------------
    # BUSCAR COMBINACIÓN MATEMÁTICA EXACTA
    #
    # cantidad = total / precio
    #
    # y cantidad debe ser múltiplo de 0,50.
    # --------------------------------------------------------

    solucion_matematica = buscar_solucion_matematica(
        precios,
        totales,
        cantidades_ocr
    )

    if solucion_matematica:

        cantidad = solucion_matematica[
            "cantidad"
        ]

        precio = solucion_matematica[
            "precio"
        ]

        total_ocr = solucion_matematica[
            "total"
        ]

        cantidad_calculada = (
            solucion_matematica[
                "cantidad_calculada"
            ]
        )

        cantidad_validada = True

        cantidad_fuente = (
            "CALCULADA_PRECIO_TOTAL"
        )

        # ----------------------------------------------------
        # ¿Tuvimos que corregir el OCR?
        # ----------------------------------------------------

        coincidencia_ocr = any(
            abs(
                cantidad_ocr
                - cantidad
            ) < 0.01
            for cantidad_ocr
            in cantidades_ocr
        )

        cantidad_forzada = (
            not coincidencia_ocr
        )

    else:

        # ----------------------------------------------------
        # NO TENEMOS PRECIO + TOTAL VALIDABLE.
        #
        # Usamos cantidad OCR solo si cumple
        # múltiplo de 0,50.
        # ----------------------------------------------------

        cantidad = (
            cantidades_ocr[0]
            if cantidades_ocr
            else None
        )

        precio = seleccionar_precio(
            precios
        )

        total_ocr = (
            totales[0]
            if totales
            else None
        )

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

    # --------------------------------------------------------
    # TOTAL COINCIDE
    # --------------------------------------------------------

    total_coincide = None

    if (
        total_ocr is not None
        and
        total_calculado is not None
    ):

        total_coincide = (
            abs(
                total_ocr
                - total_calculado
            )
            <= 1
        )

    # --------------------------------------------------------
    # DESCRIPCIÓN
    # --------------------------------------------------------

    descripcion = seleccionar_descripcion(
        codigo,
        descripciones
    )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    resultado = {

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
                        fuente
                        for fuente
                        in fuentes
                        if fuente
                    )
                )
            ),

        "fila_raw":
            " || ".join(
                fila
                for fila
                in filas
                if fila
            )
    }

    return resultado


# ============================================================
# MATEMÁTICA PRECIO / TOTAL / CANTIDAD
# ============================================================

def buscar_solucion_matematica(
    precios,
    totales,
    cantidades_ocr
):

    soluciones = []

    for precio in precios:

        if (
            precio is None
            or
            precio <= 0
        ):

            continue

        for total in totales:

            if (
                total is None
                or
                total <= 0
            ):

                continue

            # ------------------------------------------------
            # Cantidad matemática exacta aproximada
            # ------------------------------------------------

            cantidad_calculada = (
                total
                / precio
            )

            # ------------------------------------------------
            # FORZAR A PASO DE 0,50
            # ------------------------------------------------

            cantidad_normalizada = (
                normalizar_medio_m3(
                    cantidad_calculada
                )
            )

            if (
                cantidad_normalizada is None
                or
                cantidad_normalizada <= 0
                or
                cantidad_normalizada > 100
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
            # EXIGIMOS COINCIDENCIA MATEMÁTICA
            #
            # Permitimos diferencia máxima de $1
            # solo por redondeos.
            # ------------------------------------------------

            if diferencia > 1:

                continue

            # ------------------------------------------------
            # PUNTAJE
            # ------------------------------------------------

            puntaje = 1000

            # Si además OCR leyó la misma cantidad,
            # aumenta mucho la confianza.
            for cantidad_ocr in cantidades_ocr:

                if abs(
                    cantidad_ocr
                    - cantidad_normalizada
                ) < 0.01:

                    puntaje += 100

            # Preferimos precios razonables
            # para hormigón.
            if (
                20000
                <= precio
                <= 300000
            ):

                puntaje += 20

            soluciones.append({

                "precio":
                    precio,

                "total":
                    total,

                "cantidad":
                    cantidad_normalizada,

                "cantidad_calculada":
                    round(
                        cantidad_calculada,
                        6
                    ),

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
# SELECCIONAR PRECIO
# ============================================================

def seleccionar_precio(
    precios
):

    if not precios:

        return None

    # --------------------------------------------------------
    # Eliminar valores absurdamente bajos
    # --------------------------------------------------------

    razonables = [
        precio
        for precio in precios
        if (
            10000
            <= precio
            <= 500000
        )
    ]

    if razonables:

        return razonables[0]

    return precios[0]


# ============================================================
# DESCRIPCIÓN HORMIGÓN
# ============================================================

def extraer_descripcion_hormigon(
    linea
):

    # --------------------------------------------------------
    # CASO PERFECTO
    # --------------------------------------------------------

    resultado = re.search(
        r"(HORMIGON\s+.*?C/\d{1,3})",
        linea
    )

    if resultado:

        return limpiar_texto(
            resultado.group(1)
        )

    # --------------------------------------------------------
    # OCR DAÑÓ "HORMIGON"
    #
    # Intentamos conservar GR...C/XX
    # --------------------------------------------------------

    resultado = re.search(
        r"(GR\d{1,3}.*?C/\d{1,3})",
        linea
    )

    if resultado:

        return limpiar_texto(
            "HORMIGON "
            + resultado.group(1)
        )

    # --------------------------------------------------------
    # EN FASE 3 SERÁ REEMPLAZADO / VALIDADO
    # CONTRA ITEM_TED.
    # --------------------------------------------------------

    return "HORMIGON"


def seleccionar_descripcion(
    codigo,
    descripciones
):

    descripciones = [
        descripcion
        for descripcion in descripciones
        if descripcion
    ]

    if codigo == "CAR-INCOMP":

        return "CARGA INCOMPLETA"

    if not descripciones:

        return "HORMIGON"

    # --------------------------------------------------------
    # Preferir descripción que tenga:
    #
    # HORMIGON
    # +
    # C/
    # --------------------------------------------------------

    completas = [
        descripcion
        for descripcion in descripciones
        if (
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

    return max(
        descripciones,
        key=len
    )


# ============================================================
# RECORTAR TABLA
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

    texto = re.sub(
        r"\s+",
        " ",
        str(
            texto or ""
        )
    )

    return texto.strip()


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
