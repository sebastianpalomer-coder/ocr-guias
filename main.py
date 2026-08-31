import io
from io import BytesIO
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
from pytesseract import Output

app = FastAPI()

VERSION = "2.7.3-transex-facturas-rutfix"

FORMATOS_IMAGEN = {"image/jpeg", "image/jpg", "image/png"}
FORMATO_PDF = "application/pdf"

@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "service": "ocr-guias",
        "version": VERSION,
        "modulos": ["guias", "facturas"],
        "endpoints": ["/ocr", "/factura", "/factura/ping"],
        "bytesio_ok": True,
    }


@app.get("/factura/ping")
def factura_ping():
    return {
        "status": "ok",
        "modulo": "facturas",
        "version": VERSION,
        "endpoint_proceso": "/factura",
        "vinculo_guias": "REFERENCIA_DOCUMENTAL",
        "montos_usados_para_match_guia": False,
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        datos = await file.read()
        if not datos:
            raise HTTPException(status_code=400, detail="Archivo vacío")
        tipo = file.content_type or ""
        if tipo == FORMATO_PDF:
            resultado = procesar_pdf(datos)
        elif tipo in FORMATOS_IMAGEN:
            imagen = Image.open(BytesIO(datos))
            imagen = ImageOps.exif_transpose(imagen)
            resultado = procesar_imagen(imagen)
            resultado["pages"] = 1
        else:
            raise HTTPException(status_code=400, detail=f"Formato no soportado: {tipo}")
        resultado.update({"ok": True, "filename": file.filename, "content_type": tipo})
        return resultado
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/factura")
async def factura(file: UploadFile = File(...)):
    """
    Procesa facturas PDF estructuradas (iConstruye / Transex).

    La relación factura -> guía NO se determina por precios, cantidades ni montos.
    Este endpoint solamente extrae las referencias documentales de las guías que
    aparecen explícitamente en la factura.
    """
    try:
        datos = await file.read()
        if not datos:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        tipo = file.content_type or ""
        nombre = (file.filename or "").lower()
        if tipo != FORMATO_PDF and not nombre.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"/factura solo admite PDF. Formato recibido: {tipo}",
            )

        resultado = procesar_factura_pdf(datos)
        resultado.update(
            {
                "ok": True,
                "filename": file.filename,
                "content_type": tipo or FORMATO_PDF,
            }
        )
        return resultado
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


def procesar_pdf(datos: bytes):
    textos, documentos, detalles, tablas = [], [], [], []
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(datos)
        ruta_pdf = tmp.name
    try:
        pdf = pdfium.PdfDocument(ruta_pdf)
        total_paginas = len(pdf)
        for numero_pagina in range(total_paginas):
            pagina = pdf[numero_pagina]
            bitmap = pagina.render(scale=3)
            imagen = bitmap.to_pil()
            resultado_pagina = procesar_imagen(imagen)
            textos.append(f"--- PAGINA {numero_pagina + 1} ---\n{resultado_pagina['text']}")
            documentos.append(resultado_pagina["documento"])
            detalles.extend(resultado_pagina["detalle"])
            tablas.append(resultado_pagina.get("tabla_texto", ""))
        return {
            "pages": total_paginas,
            "text": "\n\n".join(textos),
            "documento": documentos[0] if documentos else {},
            "detalle": detalles,
            "tabla_texto": "\n\n".join(tablas),
        }
    finally:
        try:
            os.remove(ruta_pdf)
        except Exception:
            pass


def procesar_imagen(imagen: Image.Image):
    imagen = ImageOps.exif_transpose(imagen)
    imagen_documento = corregir_perspectiva(imagen)
    imagen_preparada = preparar_imagen(imagen_documento)

    texto_completo = ejecutar_ocr_texto(imagen_preparada, psm=6)
    documento = extraer_documento(texto_completo)

    tabla = recortar_tabla_transex(imagen_preparada)
    tabla_texto_base = ""
    if tabla is not None:
        tabla_base = preparar_recorte(tabla)
        tabla_texto_base = ejecutar_ocr_texto(tabla_base, psm=6)

    detalle = interpretar_productos_transex(
        texto_completo,
        tabla_texto_base,
        documento.get("neto_documento_ocr"),
    )

    tabla_texto = tabla_texto_base

    if detalle_necesita_rescate(detalle):
        textos_rescate = ejecutar_rescate_detalle(imagen_preparada)
        if textos_rescate:
            partes = []
            if tabla_texto_base:
                partes.append("--- OCR TABLA BASE ---\n" + tabla_texto_base)
            for indice, texto in enumerate(textos_rescate, start=1):
                partes.append(f"--- OCR RESCATE {indice} ---\n{texto}")
            tabla_texto = "\n\n".join(partes)
            detalle_rescatado = interpretar_productos_transex(
                texto_completo,
                tabla_texto,
                documento.get("neto_documento_ocr"),
            )
            if puntaje_detalle(detalle_rescatado) > puntaje_detalle(detalle):
                detalle = detalle_rescatado

    return {"text": texto_completo, "documento": documento, "detalle": detalle, "tabla_texto": tabla_texto}


def preparar_imagen(imagen: Image.Image):
    imagen = imagen.convert("L")
    imagen = ImageOps.autocontrast(imagen)
    ancho, alto = imagen.size
    ancho_objetivo = 2200
    if ancho < ancho_objetivo:
        factor = ancho_objetivo / ancho
        imagen = imagen.resize((int(ancho * factor), int(alto * factor)), Image.Resampling.LANCZOS)
    return imagen


def preparar_recorte(imagen: Image.Image):
    imagen = ImageOps.autocontrast(imagen.convert("L"))
    ancho, alto = imagen.size
    return imagen.resize((int(ancho * 1.8), int(alto * 1.8)), Image.Resampling.LANCZOS)


def ejecutar_ocr_texto(imagen: Image.Image, psm=6):
    config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(imagen, lang="spa+eng", config=config).strip()


def ejecutar_rescate_detalle(imagen: Image.Image):
    recortes = generar_recortes_rescate(imagen)
    textos = []
    for recorte in recortes:
        for variante in generar_variantes_rescate(recorte):
            for psm in (6, 11):
                texto = ejecutar_ocr_texto(variante, psm=psm).strip()
                if texto and texto not in textos:
                    textos.append(texto)
    return textos


def generar_recortes_rescate(imagen: Image.Image):
    ancho, alto = imagen.size
    recortes = []
    zona_detectada = detectar_zona_productos(imagen)
    if zona_detectada is not None:
        recortes.append(zona_detectada)
    for inicio, fin in [(0.34, 0.53), (0.37, 0.50), (0.39, 0.56)]:
        top, bottom = int(alto * inicio), int(alto * fin)
        if bottom > top:
            recortes.append(imagen.crop((0, top, ancho, bottom)))
    return recortes


def detectar_zona_productos(imagen: Image.Image):
    try:
        datos = pytesseract.image_to_data(
            imagen,
            lang="spa+eng",
            config="--oem 3 --psm 6",
            output_type=Output.DICT,
        )
    except Exception:
        return None

    tops_inicio, tops_fin = [], []
    cantidad = len(datos.get("text", []))
    for i in range(cantidad):
        texto = normalizar_texto(datos["text"][i])
        if not texto:
            continue
        top = int(datos["top"][i])
        height = int(datos["height"][i])
        if "OBSERVACIONES" in texto or texto == "OBSERVACION":
            tops_inicio.append(top + height)
        if texto in {"ITEM", "SUBTOTAL", "NETO"}:
            tops_fin.append(top)

    if not tops_inicio:
        return None

    top = min(tops_inicio)
    candidatos_fin = [y for y in tops_fin if y > top]
    bottom = min(candidatos_fin) if candidatos_fin else min(imagen.height, top + int(imagen.height * 0.18))
    margen = int(imagen.height * 0.01)
    top = max(0, top - margen)
    bottom = min(imagen.height, bottom + margen)
    if bottom <= top:
        return None
    return imagen.crop((0, top, imagen.width, bottom))


def generar_variantes_rescate(imagen: Image.Image):
    base = preparar_recorte(imagen)
    gris = np.array(base.convert("L"))
    variantes = [base]

    _, otsu = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(Image.fromarray(otsu))

    adaptativa = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    variantes.append(Image.fromarray(adaptativa))

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    enfocada = cv2.filter2D(gris, -1, kernel)
    variantes.append(Image.fromarray(enfocada))
    return variantes


def corregir_perspectiva(imagen: Image.Image):
    rgb = np.array(imagen.convert("RGB"))
    original = rgb.copy()
    alto, ancho = rgb.shape[:2]
    escala = 1200 / max(alto, ancho)
    if escala < 1:
        pequena = cv2.resize(rgb, None, fx=escala, fy=escala)
    else:
        pequena = rgb.copy()
        escala = 1

    gris = cv2.cvtColor(pequena, cv2.COLOR_RGB2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(gris, 50, 150)
    bordes = cv2.dilate(bordes, None, iterations=1)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

    pagina = None
    area_imagen = pequena.shape[0] * pequena.shape[1]
    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        aproximado = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        if len(aproximado) != 4:
            continue
        area = cv2.contourArea(aproximado)
        if area < area_imagen * 0.45:
            continue
        pagina = aproximado.reshape(4, 2).astype(np.float32)
        break

    if pagina is None:
        return imagen

    pagina = pagina / escala
    ordenados = ordenar_puntos(pagina)
    tl, tr, br, bl = ordenados
    ancho_final = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    alto_final = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    if ancho_final < 500 or alto_final < 500:
        return imagen

    destino = np.array(
        [[0, 0], [ancho_final - 1, 0], [ancho_final - 1, alto_final - 1], [0, alto_final - 1]],
        dtype=np.float32,
    )
    matriz = cv2.getPerspectiveTransform(ordenados, destino)
    corregida = cv2.warpPerspective(original, matriz, (ancho_final, alto_final))
    return Image.fromarray(corregida)


def ordenar_puntos(puntos):
    rect = np.zeros((4, 2), dtype=np.float32)
    suma = puntos.sum(axis=1)
    rect[0] = puntos[np.argmin(suma)]
    rect[2] = puntos[np.argmax(suma)]
    diferencia = np.diff(puntos, axis=1).flatten()
    rect[1] = puntos[np.argmin(diferencia)]
    rect[3] = puntos[np.argmax(diferencia)]
    return rect


def extraer_documento(texto):
    total_documento = extraer_total_documento(texto)
    neto_documento = extraer_neto_documento(texto)

    if neto_documento is None and total_documento is not None:
        neto_estimado = round(total_documento / 1.19)
        iva_estimado = round(neto_estimado * 0.19)
        if neto_estimado + iva_estimado == total_documento:
            neto_documento = neto_estimado

    return {
        "folio_ocr": extraer_folio(texto),
        "folio_parcial_ocr": extraer_folio_parcial(texto),
        "fecha_emision_ocr": extraer_fecha_emision(texto),
        "rut_emisor_ocr": extraer_rut_emisor(texto),
        "neto_documento_ocr": neto_documento,
        "total_documento_ocr": total_documento,
        "obra": extraer_obra(texto),
        "patente": extraer_patente(texto),
    }


def extraer_folio(texto):
    candidatos = re.findall(r"N\s*[°º]?\s*(\d{6})\b", texto.upper())
    if not candidatos:
        return None
    frecuencias = {}
    for numero in candidatos:
        frecuencias[numero] = frecuencias.get(numero, 0) + 1
    return max(frecuencias, key=frecuencias.get)


def extraer_folio_parcial(texto):
    resultado = re.search(r"\bN\s*[°º]?\s*(\d{5})\b", texto.upper())
    return resultado.group(1) if resultado else None


def extraer_fecha_emision(texto):
    resultado = re.search(
        r"FECHA\s+EMISION\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})",
        texto.upper(),
    )
    if resultado:
        fecha = validar_fecha(resultado.group(1))
        if fecha:
            return fecha

    candidatos = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", texto)
    fechas_objeto = []
    for candidato in candidatos:
        fecha = validar_fecha(candidato)
        if fecha:
            try:
                fechas_objeto.append((datetime.strptime(fecha, "%d/%m/%Y"), fecha))
            except Exception:
                pass
    if not fechas_objeto:
        return None
    fechas_objeto.sort(key=lambda x: x[0])
    return fechas_objeto[0][1]


def validar_fecha(texto):
    try:
        return datetime.strptime(texto, "%d/%m/%Y").strftime("%d/%m/%Y")
    except Exception:
        return None


def extraer_rut_emisor(texto):
    """
    Extrae el RUT del emisor de una guía.

    IMPORTANTE:
    - Las fotografías pueden alterar el orden visual de las columnas.
    - En guías Transex se ha observado que Tesseract puede separar
      "88.147." de "600-2" y luego confundir un teléfono (por ejemplo
      28026120) con un RUT.
    - Por eso:
        1) se reconoce explícitamente al proveedor Transex;
        2) cualquier candidato genérico debe pasar validación de dígito
           verificador chileno antes de ser aceptado.
    """
    texto_upper = str(texto or "").upper()
    texto_compacto = re.sub(r"\s+", " ", texto_upper)
    solo_alfanum = re.sub(r"[^A-Z0-9]", "", texto_upper)

    # Proveedor específico del MVP.
    # El OCR puede leer el encabezado en distinto orden, pero mientras
    # identifique HORMIGONES + TRANSEX podemos recuperar de forma segura
    # el RUT canónico del emisor.
    if "HORMIGONES" in texto_upper and "TRANSEX" in texto_upper:
        return "88147600-2"

    # Lectura directa normal del RUT conocido.
    if re.search(
        r"88[\.\s]*147[\.\s]*600[\-\s]*2",
        texto_upper,
    ):
        return "88147600-2"

    # Primero intenta candidatos cercanos a etiquetas RUT.
    patrones_contexto = [
        r"R\.?\s*U\.?\s*T\.?\s*[:\-]?\s*([0-9\.\s\-K]{8,20})",
        r"\bRUT\s*[:\-]?\s*([0-9\.\s\-K]{8,20})",
    ]

    for patron in patrones_contexto:
        for coincidencia in re.findall(patron, texto_upper, re.IGNORECASE):
            candidato = normalizar_rut(coincidencia)
            if candidato and validar_rut_chileno(candidato):
                return candidato

    # Fallback genérico: nunca aceptar un número solo por tener forma de RUT.
    # Debe tener DV válido; esto evita confundir teléfonos con RUT.
    candidatos = re.findall(
        r"\b\d{1,2}[\.\s]?\d{3}[\.\s]?\d{3}[\-\s]?[0-9K]\b",
        texto_upper,
    )

    for candidato_raw in candidatos:
        candidato = normalizar_rut(candidato_raw)
        if candidato and validar_rut_chileno(candidato):
            return candidato

    return None


def normalizar_rut(rut):
    limpio = re.sub(r"[^0-9K]", "", str(rut or "").upper())
    if len(limpio) < 2:
        return None
    return limpio[:-1] + "-" + limpio[-1]


def validar_rut_chileno(rut):
    """
    Valida el dígito verificador de un RUT chileno.
    Retorna True solo si el RUT es matemáticamente válido.
    """
    normalizado = normalizar_rut(rut)
    if not normalizado:
        return False

    cuerpo, dv = normalizado.split("-")

    if not cuerpo.isdigit() or not cuerpo:
        return False

    suma = 0
    multiplicador = 2

    for digito in reversed(cuerpo):
        suma += int(digito) * multiplicador
        multiplicador += 1
        if multiplicador > 7:
            multiplicador = 2

    resultado = 11 - (suma % 11)

    if resultado == 11:
        dv_calculado = "0"
    elif resultado == 10:
        dv_calculado = "K"
    else:
        dv_calculado = str(resultado)

    return dv_calculado == dv.upper()


def extraer_neto_documento(texto):
    candidatos = []
    for linea in texto.splitlines():
        linea_upper = linea.upper()
        if "SUBTOTAL" in linea_upper:
            continue
        if not re.search(r"\bNETO\b", linea_upper):
            continue
        for monto in extraer_montos(linea):
            if monto >= 1000:
                candidatos.append(monto)
    return max(candidatos) if candidatos else None


def extraer_total_documento(texto):
    candidatos = []
    for linea in texto.splitlines():
        linea_upper = linea.upper()
        if "SUBTOTAL" in linea_upper:
            continue
        if not re.search(r"\bTOTAL\b", linea_upper):
            continue
        for monto in extraer_montos(linea):
            if monto >= 1000:
                candidatos.append(monto)
    return max(candidatos) if candidatos else None


def extraer_obra(texto):
    texto_upper = texto.upper()
    if "HOTEL BELLET" in texto_upper:
        return "HOTEL BELLET"
    resultado = re.search(r"OBRA\s*[:|*]?\s*([A-Z0-9 \-]{3,40})", texto_upper)
    if resultado:
        obra = re.split(r"[\n\r|\\]", resultado.group(1))[0]
        obra = re.sub(r"\s+", " ", obra).strip()
        if len(obra) >= 3:
            return obra
    return None


def extraer_patente(texto):
    texto_upper = texto.upper()
    patrones = [
        r"PATENTE[\s:>\-|]*([A-Z]{3,4}[\-\s]?\d{2})",
        r"PATENTE[\s:>\-|]*([A-Z]{2}[\-\s]?\d{4})",
    ]
    for patron in patrones:
        resultado = re.search(patron, texto_upper)
        if resultado:
            return normalizar_patente(resultado.group(1))
    return None


def normalizar_patente(patente):
    patente = str(patente or "").upper().strip().replace(" ", "")
    resultado = re.fullmatch(r"([A-Z]{3,4})-?(\d{2})", patente)
    if resultado:
        return resultado.group(1) + "-" + resultado.group(2)
    resultado = re.fullmatch(r"([A-Z]{2})-?(\d{4})", patente)
    if resultado:
        return resultado.group(1) + "-" + resultado.group(2)
    return patente


def normalizar_medio_m3(cantidad):
    if cantidad is None:
        return None
    try:
        cantidad = float(cantidad)
    except Exception:
        return None
    if cantidad <= 0:
        return None
    return round(cantidad * 2) / 2


def cantidad_es_multiplo_medio(cantidad):
    if cantidad is None:
        return False
    normalizada = normalizar_medio_m3(cantidad)
    return normalizada is not None and abs(float(cantidad) - normalizada) < 0.001


def interpretar_productos_transex(texto_completo, tabla_texto, neto_documento=None):
    candidatos = []
    candidatos.extend(buscar_productos_en_texto(texto_completo, "TEXTO_OCR"))
    candidatos.extend(buscar_productos_en_texto(tabla_texto, "TABLA_OCR"))

    agrupados = {}
    for candidato in candidatos:
        agrupados.setdefault(candidato["codigo"], []).append(candidato)

    if len(agrupados) == 1 and "4489" in agrupados and neto_documento is not None:
        for opcion in agrupados["4489"]:
            opcion["neto_documento"] = neto_documento

    resultados = []
    for codigo, opciones in agrupados.items():
        resultado = resolver_producto(codigo, opciones)
        if resultado:
            resultados.append(resultado)
    return resultados


def generar_bloques(texto):
    lineas = [re.sub(r"\s+", " ", linea).strip() for linea in texto.splitlines()]
    bloques = []
    for indice, linea in enumerate(lineas):
        if not linea:
            continue
        tipo = detectar_tipo_producto_linea(linea)
        if not tipo:
            continue
        partes = [linea]
        for offset in (1, 2):
            posicion = indice + offset
            if posicion >= len(lineas):
                break
            siguiente = lineas[posicion]
            if not siguiente:
                continue
            if detectar_tipo_producto_linea(siguiente):
                break
            siguiente_upper = siguiente.upper()
            if any(
                marcador in siguiente_upper
                for marcador in (
                    "ITEM CANTIDAD",
                    "SOLICITANTE",
                    "SUBTOTAL",
                    "DESCUENTO",
                    "NETO",
                    "I.V.A",
                    "IVA (",
                    "TOTAL ",
                )
            ):
                break
            partes.append(siguiente)
        bloques.append({"tipo": tipo, "texto": " ".join(partes)})
    return bloques


def detectar_tipo_producto_linea(linea):
    texto = linea.upper()
    if "CARGA" in texto and ("INCOMP" in texto or "INCO" in texto):
        return "CARGA"
    if re.search(r"\b4489\b", texto):
        return "HORMIGON"
    if "C/" in texto and (
        "HORMIGON" in texto
        or "ORMIGON" in texto
        or re.search(r"GR\d{1,3}", texto)
    ):
        return "HORMIGON"
    return None


def buscar_productos_en_texto(texto, fuente):
    resultados = []
    if not texto:
        return resultados
    for bloque in generar_bloques(texto):
        if bloque["tipo"] == "HORMIGON":
            item = interpretar_bloque_hormigon(bloque["texto"], fuente)
        else:
            item = interpretar_bloque_carga(bloque["texto"], fuente)
        if item:
            resultados.append(item)
    return resultados


def interpretar_bloque_hormigon(texto, fuente):
    texto_upper = texto.upper()
    if not re.search(r"\b4489\b", texto_upper) and not (
        "C/" in texto_upper
        and (
            "HORMIGON" in texto_upper
            or "ORMIGON" in texto_upper
            or re.search(r"GR\d{1,3}", texto_upper)
        )
    ):
        return None

    descripcion = extraer_descripcion_hormigon(texto_upper)
    resultado = re.search(r"C/\d{1,3}", texto_upper)
    if resultado:
        cola = texto[resultado.end():]
    else:
        resultado_codigo = re.search(r"\b4489\b", texto_upper)
        cola = texto[resultado_codigo.end():] if resultado_codigo else texto

    componentes = extraer_componentes_producto(cola)
    return {
        "codigo": "4489",
        "descripcion": descripcion,
        "unidad": "M3",
        "cantidad_ocr": componentes["cantidad_ocr"],
        "montos": componentes["montos"],
        "fuente": fuente,
        "fila_raw": texto,
    }


def interpretar_bloque_carga(texto, fuente):
    texto_upper = texto.upper()
    resultado = re.search(r"CARGA\s+INCO[A-Z]*", texto_upper)
    cola = texto[resultado.end():] if resultado else texto
    componentes = extraer_componentes_producto(cola)
    return {
        "codigo": "CAR-INCOMP",
        "descripcion": "CARGA INCOMPLETA",
        "unidad": "M3",
        "cantidad_ocr": componentes["cantidad_ocr"],
        "montos": componentes["montos"],
        "fuente": fuente,
        "fila_raw": texto,
    }


def extraer_componentes_producto(texto):
    cantidades_raw = re.findall(r"(?<!\d)(\d{1,3}[,.](?:00|50))(?!\d)", texto)
    cantidades = []
    for raw in cantidades_raw:
        cantidad = convertir_decimal(raw)
        if cantidad is not None and 0 < cantidad <= 100 and cantidad_es_multiplo_medio(cantidad):
            cantidades.append(cantidad)

    resultado_m3_entero = re.search(r"\bM3\b[\s|:\-]*(\d{1,2})\b", texto.upper())
    if resultado_m3_entero:
        cantidad = float(resultado_m3_entero.group(1))
        if 0 < cantidad <= 100:
            cantidades.append(cantidad)

    cantidades = lista_unica(cantidades)
    return {
        "cantidad_ocr": cantidades[0] if cantidades else None,
        "cantidades_ocr": cantidades,
        "montos": extraer_montos(texto),
    }


def extraer_montos(texto):
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
    for coincidencia in patron.finditer(texto):
        valor = convertir_monto(coincidencia.group(1))
        if valor is not None and 1000 <= valor <= 9999999:
            resultados.append(valor)
    return resultados


def resolver_producto(codigo, opciones):
    cantidades_ocr, todos_montos, fuentes, filas, descripciones, netos_documento = [], [], [], [], [], []
    for opcion in opciones:
        cantidad = opcion.get("cantidad_ocr")
        if cantidad is not None:
            cantidades_ocr.append(cantidad)
        todos_montos.extend(opcion.get("montos", []))
        if opcion.get("fuente"):
            fuentes.append(opcion["fuente"])
        if opcion.get("fila_raw"):
            filas.append(opcion["fila_raw"])
        if opcion.get("descripcion"):
            descripciones.append(opcion["descripcion"])
        if opcion.get("neto_documento") is not None:
            netos_documento.append(opcion["neto_documento"])

    cantidades_ocr = lista_unica(cantidades_ocr)
    montos_unicos = lista_unica(todos_montos)

    solucion = buscar_solucion_matematica(montos_unicos, cantidades_ocr)
    if solucion is None and netos_documento:
        solucion = buscar_solucion_con_neto(
            montos_unicos,
            cantidades_ocr,
            netos_documento[0],
        )

    if solucion:
        cantidad = solucion["cantidad"]
        precio = solucion["precio"]
        total_ocr = solucion["total"]
        cantidad_calculada = solucion.get("cantidad_calculada")
        cantidad_validada = True
        cantidad_fuente = solucion.get("cantidad_fuente", "CALCULADA_PRECIO_TOTAL")
        coincide_con_ocr = any(abs(valor - cantidad) < 0.001 for valor in cantidades_ocr)
        cantidad_forzada = not coincide_con_ocr
    else:
        cantidad = cantidades_ocr[0] if cantidades_ocr else None
        precio = seleccionar_precio(todos_montos)
        total_ocr = None
        cantidad_calculada = None
        cantidad_validada = False
        cantidad_fuente = "OCR" if cantidad is not None else None
        cantidad_forzada = False

    total_calculado = round(cantidad * precio) if cantidad is not None and precio is not None else None
    total_coincide = (
        abs(total_calculado - total_ocr) <= 1
        if total_ocr is not None and total_calculado is not None
        else None
    )

    return {
        "codigo": codigo,
        "descripcion": seleccionar_descripcion(codigo, descripciones),
        "unidad": "M3",
        "cantidad_ocr": cantidades_ocr[0] if cantidades_ocr else None,
        "cantidad_calculada": cantidad_calculada,
        "cantidad": cantidad,
        "cantidad_fuente": cantidad_fuente,
        "cantidad_forzada": cantidad_forzada,
        "cantidad_validada": cantidad_validada,
        "regla_cantidad": "MULTIPLO_0_50",
        "precio": precio,
        "total_ocr": total_ocr,
        "total_calculado": total_calculado,
        "total_coincide": total_coincide,
        "parser_ok": cantidad is not None and precio is not None,
        "fuente": "+".join(sorted(set(fuentes))),
        "montos_detectados": montos_unicos,
        "fila_raw": " || ".join(filas),
    }


def buscar_solucion_matematica(montos, cantidades_ocr):
    soluciones = []
    for precio in montos:
        if not 10000 <= precio <= 500000:
            continue
        for total in montos:
            if total <= precio:
                continue
            cantidad_exacta = total / precio
            cantidad_normalizada = normalizar_medio_m3(cantidad_exacta)
            if cantidad_normalizada is None or not 0.5 <= cantidad_normalizada <= 100:
                continue
            total_validacion = round(cantidad_normalizada * precio)
            diferencia = abs(total_validacion - total)
            if diferencia > 1:
                continue
            puntaje = 1000
            for cantidad_ocr in cantidades_ocr:
                if abs(cantidad_ocr - cantidad_normalizada) < 0.001:
                    puntaje += 500
            if 30000 <= precio <= 200000:
                puntaje += 50
            soluciones.append(
                {
                    "cantidad": cantidad_normalizada,
                    "cantidad_calculada": round(cantidad_exacta, 6),
                    "cantidad_fuente": "CALCULADA_PRECIO_TOTAL",
                    "precio": precio,
                    "total": total,
                    "diferencia": diferencia,
                    "puntaje": puntaje,
                }
            )
    if not soluciones:
        return None
    soluciones.sort(key=lambda x: (x["puntaje"], -x["diferencia"]), reverse=True)
    return soluciones[0]


def buscar_solucion_con_neto(montos, cantidades_ocr, neto_documento):
    """
    Fallback exclusivo para documentos con un solo producto detectado.

    Usa NETO_DOCUMENTO como total de línea solo cuando:
    - existe un precio OCR que cuadra exactamente con una cantidad en pasos de 0,50; o
    - no existe ningún precio OCR razonable y sí existe una cantidad OCR válida,
      en cuyo caso calcula el precio = neto / cantidad.

    Si hay un precio OCR razonable pero no cuadra con el neto, NO inventa otro precio.
    Esto evita falsos positivos cuando falta una segunda línea como CARGA INCOMPLETA.
    """

    if neto_documento is None or neto_documento <= 0:
        return None

    precios_razonables = [
        monto
        for monto in montos
        if 10000 <= monto <= 500000 and monto < neto_documento
    ]

    soluciones = []

    for precio in precios_razonables:
        cantidad_exacta = neto_documento / precio
        cantidad_normalizada = normalizar_medio_m3(cantidad_exacta)

        if cantidad_normalizada is None or not 0.5 <= cantidad_normalizada <= 100:
            continue

        if abs(round(cantidad_normalizada * precio) - neto_documento) > 1:
            continue

        puntaje = 1200
        if any(abs(c - cantidad_normalizada) < 0.001 for c in cantidades_ocr):
            puntaje += 500
        if 30000 <= precio <= 200000:
            puntaje += 50

        soluciones.append({
            "cantidad": cantidad_normalizada,
            "cantidad_calculada": round(cantidad_exacta, 6),
            "cantidad_fuente": "CALCULADA_PRECIO_NETO",
            "precio": precio,
            "total": neto_documento,
            "diferencia": abs(round(cantidad_normalizada * precio) - neto_documento),
            "puntaje": puntaje,
        })

    if soluciones:
        soluciones.sort(key=lambda x: (x["puntaje"], -x["diferencia"]), reverse=True)
        return soluciones[0]

    # Si OCR vio un precio razonable, pero no cuadra con el neto, no calculamos
    # un precio alternativo a partir de la cantidad.
    if precios_razonables:
        return None

    for cantidad in cantidades_ocr:
        if not cantidad_es_multiplo_medio(cantidad):
            continue

        precio_exacto = neto_documento / cantidad
        precio_redondeado = round(precio_exacto)

        if abs(precio_exacto - precio_redondeado) > 0.001:
            continue

        if not 10000 <= precio_redondeado <= 500000:
            continue

        if round(cantidad * precio_redondeado) != neto_documento:
            continue

        return {
            "cantidad": cantidad,
            "cantidad_calculada": cantidad,
            "cantidad_fuente": "OCR_CANTIDAD_NETO",
            "precio": precio_redondeado,
            "total": neto_documento,
            "diferencia": 0,
            "puntaje": 1000,
        }

    return None

def seleccionar_precio(montos):
    candidatos = [monto for monto in montos if 10000 <= monto <= 200000]
    if not candidatos:
        return None
    return max(set(candidatos), key=candidatos.count)


def extraer_descripcion_hormigon(texto):
    resultado = re.search(r"(HORMIGON\s+.*?C/\d{1,3})", texto)
    if resultado:
        return limpiar_texto(resultado.group(1))
    resultado = re.search(r"(GR\d{1,3}.*?C/\d{1,3})", texto)
    if resultado:
        return limpiar_texto("HORMIGON " + resultado.group(1))
    if "C/08" in texto:
        return "HORMIGON GR20-90%-40 C/08"
    return "HORMIGON"


def seleccionar_descripcion(codigo, descripciones):
    if codigo == "CAR-INCOMP":
        return "CARGA INCOMPLETA"
    completas = [
        descripcion
        for descripcion in descripciones
        if descripcion and "HORMIGON" in descripcion.upper() and "C/" in descripcion.upper()
    ]
    return max(completas, key=len) if completas else "HORMIGON"


def recortar_tabla_transex(imagen):
    zona = detectar_zona_productos(imagen)
    if zona is not None:
        return zona
    ancho, alto = imagen.size
    top, bottom = int(alto * 0.36), int(alto * 0.58)
    if bottom <= top:
        return None
    return imagen.crop((0, top, ancho, bottom))


def detalle_necesita_rescate(detalle):
    if not detalle:
        return True
    return any(not item.get("parser_ok", False) for item in detalle)


def puntaje_detalle(detalle):
    if not detalle:
        return 0
    puntaje = 0
    for item in detalle:
        if item.get("parser_ok"):
            puntaje += 10
        if item.get("cantidad") is not None:
            puntaje += 3
        if item.get("precio") is not None:
            puntaje += 3
        if item.get("total_calculado") is not None:
            puntaje += 2
        if item.get("cantidad_validada"):
            puntaje += 4
        if item.get("total_coincide"):
            puntaje += 4
    return puntaje


def convertir_decimal(texto):
    if texto is None:
        return None
    try:
        return float(str(texto).replace(",", "."))
    except Exception:
        return None


def convertir_monto(texto):
    if texto is None:
        return None
    valor = re.sub(r"[^0-9]", "", str(texto))
    if not valor:
        return None
    try:
        return int(valor)
    except Exception:
        return None


def lista_unica(valores):
    resultado = []
    for valor in valores:
        if valor not in resultado:
            resultado.append(valor)
    return resultado


def limpiar_texto(texto):
    return re.sub(r"\s+", " ", str(texto or "")).strip()


def normalizar_texto(texto):
    texto = unicodedata.normalize("NFD", str(texto or "").upper())
    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )
    return re.sub(r"[^A-Z0-9]", "", texto)


# =============================================================================
# FACTURAS ESTRUCTURADAS iCONSTRUYE / TRANSEX
# Versión agregada en 2.7
# =============================================================================

def procesar_factura_pdf(datos: bytes):
    """
    Extrae una factura estructurada desde PDF.

    Prioridad:
      1) texto embebido del PDF (sin OCR);
      2) OCR de respaldo solo si el PDF no contiene texto suficiente.

    IMPORTANTE:
    - Las guías referenciadas se extraen únicamente de la sección documental
      "GUÍA DE DESPACHO ELECTRÓNICA".
    - Los montos de factura nunca se usan para decidir qué guía corresponde.
    """
    texto, paginas, fuente_texto = extraer_texto_factura_pdf(datos)
    if not texto or len(texto.strip()) < 20:
        raise ValueError("No fue posible obtener texto utilizable desde la factura")

    documento = extraer_documento_factura(texto)
    guias = extraer_guias_referenciadas_factura(texto)
    detalle = extraer_detalle_factura(texto)

    id_factura = None
    if documento.get("rut_emisor") and documento.get("folio_factura"):
        id_factura = (
            f"{documento['rut_emisor']}-"
            f"{documento.get('tipo_dte', 33)}-"
            f"{documento['folio_factura']}"
        )
    documento["id_factura"] = id_factura

    suma_detalle = sum(
        int(item.get("total_linea") or 0)
        for item in detalle
        if item.get("incluir_en_neto", False)
    )
    neto = documento.get("neto_factura")
    total = documento.get("total_factura")
    iva = documento.get("iva_factura")

    validacion = {
        "cantidad_guias_referenciadas": len(guias),
        "cantidad_lineas_producto": sum(
            1 for item in detalle if item.get("tipo_linea") == "PRODUCTO"
        ),
        "suma_detalle_factura": suma_detalle,
        "detalle_vs_neto_coincide": (
            suma_detalle == neto if neto is not None and suma_detalle > 0 else None
        ),
        "neto_mas_iva_vs_total_coincide": (
            neto + iva == total
            if neto is not None and iva is not None and total is not None
            else None
        ),
        "regla_vinculo_guias": "REFERENCIA_DOCUMENTAL",
        "montos_usados_para_match_guia": False,
    }

    return {
        "pages": paginas,
        "text": texto,
        "fuente_texto": fuente_texto,
        "documento": documento,
        "guias_referenciadas": guias,
        "detalle": detalle,
        "validacion": validacion,
    }


def extraer_texto_factura_pdf(datos: bytes):
    """Extrae texto embebido; usa OCR solamente como respaldo."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(datos)
        ruta_pdf = tmp.name

    try:
        pdf = pdfium.PdfDocument(ruta_pdf)
        total_paginas = len(pdf)
        textos_embebidos = []

        for numero_pagina in range(total_paginas):
            pagina = pdf[numero_pagina]
            try:
                textpage = pagina.get_textpage()
                texto = textpage.get_text_range() or ""
            except Exception:
                texto = ""
            textos_embebidos.append(texto.strip())

        texto_directo = "\n\n".join(
            texto for texto in textos_embebidos if texto
        ).strip()

        # Los PDF de iConstruye normalmente superan ampliamente este umbral.
        if len(texto_directo) >= 100:
            return texto_directo, total_paginas, "PDF_TEXTO"

        # Fallback OCR para un eventual PDF escaneado.
        textos_ocr = []
        for numero_pagina in range(total_paginas):
            pagina = pdf[numero_pagina]
            bitmap = pagina.render(scale=3)
            imagen = bitmap.to_pil()
            imagen = ImageOps.exif_transpose(imagen)
            preparada = preparar_imagen(imagen)
            texto_ocr = ejecutar_ocr_texto(preparada, psm=6)
            textos_ocr.append(
                f"--- PAGINA {numero_pagina + 1} ---\n{texto_ocr}"
            )

        return "\n\n".join(textos_ocr).strip(), total_paginas, "OCR_FALLBACK"
    finally:
        try:
            os.remove(ruta_pdf)
        except Exception:
            pass


def extraer_documento_factura(texto):
    proveedor = extraer_proveedor_factura(texto)
    rut_emisor = extraer_rut_emisor_factura(texto)
    folio = extraer_folio_factura(texto)

    return {
        "rut_emisor": rut_emisor,
        "proveedor": proveedor,
        "tipo_dte": 33 if detectar_factura_electronica(texto) else None,
        "folio_factura": folio,
        "fecha_emision": extraer_fecha_factura_etiqueta(
            texto, r"FECHA\s+EMISI[ÓO]N"
        ),
        "rut_receptor": extraer_rut_receptor_factura(texto),
        "razon_social_receptor": extraer_razon_social_receptor_factura(texto),
        "forma_pago": extraer_forma_pago_factura(texto),
        "fecha_vencimiento": extraer_fecha_factura_etiqueta(
            texto, r"FECHA\s+VENCIMIENTO"
        ),
        "orden_compra": extraer_orden_compra_factura(texto),
        "neto_factura": extraer_monto_factura_etiqueta(
            texto, r"TOTAL\s+NETO"
        ),
        "exento_factura": extraer_monto_factura_etiqueta(
            texto, r"TOTAL\s+EXENTO"
        ),
        "iva_factura": extraer_monto_factura_etiqueta(
            texto, r"TOTAL\s+I\.?V\.?A\.?\s*\(\s*19\s*%\s*\)"
        ),
        "total_factura": extraer_monto_factura_etiqueta(
            texto, r"MONTO\s+TOTAL"
        ),
    }


def detectar_factura_electronica(texto):
    normalizado = quitar_acentos_factura(texto).upper()
    return "FACTURA ELECTRONICA" in normalizado


def extraer_proveedor_factura(texto):
    lineas = [limpiar_texto(linea) for linea in str(texto or "").splitlines()]
    for linea in lineas:
        if not linea:
            continue
        superior = quitar_acentos_factura(linea).upper()
        if "FACTURA" in superior:
            continue
        if "R.U.T" in superior or "RUT" == superior:
            continue
        # En los PDF de iConstruye/Transex la primera línea corresponde al emisor.
        return linea
    return None


def extraer_rut_emisor_factura(texto):
    resultado = re.search(
        r"R\.?\s*U\.?\s*T\.?\s*:\s*([0-9\.\-Kk]+)",
        str(texto or ""),
        re.IGNORECASE,
    )
    if resultado:
        return normalizar_rut(resultado.group(1))
    return None


def extraer_folio_factura(texto):
    resultado = re.search(
        r"FACTURA\s+ELECTR[ÓO]NICA\s*(?:\r?\n|\s)*"
        r"N\s*[°º]?\s*(\d{1,12})",
        str(texto or ""),
        re.IGNORECASE,
    )
    if not resultado:
        return None
    try:
        return int(resultado.group(1))
    except Exception:
        return None


def extraer_fecha_factura_etiqueta(texto, etiqueta_regex):
    resultado = re.search(
        etiqueta_regex + r"\s*:\s*(\d{2}[-/]\d{2}[-/]\d{4})",
        str(texto or ""),
        re.IGNORECASE,
    )
    return normalizar_fecha_factura(resultado.group(1)) if resultado else None


def normalizar_fecha_factura(valor):
    if not valor:
        return None
    candidato = str(valor).strip().replace("/", "-")
    try:
        return datetime.strptime(candidato, "%d-%m-%Y").strftime("%d/%m/%Y")
    except Exception:
        return None


def extraer_rut_receptor_factura(texto):
    # Se exige que la línea comience en "Rut :" para no tomar el RUT del emisor.
    resultado = re.search(
        r"^\s*RUT\s*:\s*([0-9\.\-Kk]+)",
        str(texto or ""),
        re.IGNORECASE | re.MULTILINE,
    )
    return normalizar_rut(resultado.group(1)) if resultado else None


def extraer_razon_social_receptor_factura(texto):
    resultado = re.search(
        r"SEÑOR\s*\(ES\)\s*:\s*(.*?)\s+CIUDAD\s*:",
        str(texto or ""),
        re.IGNORECASE,
    )
    return limpiar_texto(resultado.group(1)) if resultado else None


def extraer_forma_pago_factura(texto):
    resultado = re.search(
        r"FORMA\s+DE\s+PAGO\s*:\s*([^\r\n]+)",
        str(texto or ""),
        re.IGNORECASE,
    )
    return limpiar_texto(resultado.group(1)) if resultado else None


def extraer_orden_compra_factura(texto):
    resultado = re.search(
        r"ORDEN\s+DE\s+COMPRA\s+([A-Z0-9\-]+)"
        r"(?:\s+\d{2}[-/]\d{2}[-/]\d{4})?",
        str(texto or ""),
        re.IGNORECASE,
    )
    return resultado.group(1).strip() if resultado else None


def extraer_monto_factura_etiqueta(texto, etiqueta_regex):
    resultado = re.search(
        etiqueta_regex + r"\s*:\s*\$?\s*([0-9\.\s]+)",
        str(texto or ""),
        re.IGNORECASE,
    )
    return convertir_monto_factura(resultado.group(1)) if resultado else None


def convertir_monto_factura(valor):
    if valor is None:
        return None
    limpio = re.sub(r"[^0-9]", "", str(valor))
    if not limpio:
        return None
    try:
        return int(limpio)
    except Exception:
        return None


def convertir_decimal_factura(valor):
    if valor is None:
        return None
    limpio = str(valor).strip().replace(".", "").replace(",", ".")
    try:
        return float(limpio)
    except Exception:
        return None


def quitar_acentos_factura(texto):
    normalizado = unicodedata.normalize("NFD", str(texto or ""))
    return "".join(
        caracter
        for caracter in normalizado
        if unicodedata.category(caracter) != "Mn"
    )


def extraer_guias_referenciadas_factura(texto):
    """
    Extrae exclusivamente referencias documentales explícitas a Guías de Despacho.
    No compara montos, cantidades, descripción ni precio.
    """
    texto_lineal = re.sub(r"\s+", " ", str(texto or "")).strip()
    patron = re.compile(
        r"GU[IÍ]A\s+DE\s+DESPACHO\s+ELECTR[ÓO]NICA\s+"
        r"(\d{5,12})\s+(\d{2}[-/]\d{2}[-/]\d{4})",
        re.IGNORECASE,
    )

    resultados = []
    folios_vistos = set()
    for coincidencia in patron.finditer(texto_lineal):
        folio = coincidencia.group(1)
        if folio in folios_vistos:
            continue
        folios_vistos.add(folio)
        resultados.append(
            {
                "folio_guia": int(folio),
                "fecha_guia": normalizar_fecha_factura(coincidencia.group(2)),
                "tipo_referencia": "GUIA_DESPACHO",
                "fuente_vinculo": "REFERENCIA_DOCUMENTAL_PDF",
            }
        )

    return resultados


def extraer_detalle_factura(texto):
    """
    Extrae líneas económicas de la propia factura.

    Estas líneas sirven para registrar lo facturado y validar internamente la factura.
    NO se utilizan para vincular una línea con una guía de despacho.
    """
    texto = str(texto or "")
    patron_producto = re.compile(
        r"(?P<cantidad>\d{1,3}[,.]\d{4})\s*(?:00\s*)?"
        r"(?P<codigo>\d{3,12})\s+"
        r"(?P<descripcion1>[^\r\n]+)\s*[\r\n]+"
        r"(?P<descripcion2>[^\r\n]+)\s*[\r\n]+"
        r"\$\s*(?P<precio>[\d\.]+),\d{2}\s+"
        r"\$\s*(?P<total>[\d\.]+)",
        re.IGNORECASE,
    )

    detalle = []
    numero_linea = 1
    for coincidencia in patron_producto.finditer(texto):
        descripcion1 = limpiar_texto(coincidencia.group("descripcion1"))
        descripcion2 = limpiar_texto(coincidencia.group("descripcion2"))
        descripcion = descripcion1 or descripcion2

        detalle.append(
            {
                "nro_linea": numero_linea,
                "codigo": coincidencia.group("codigo"),
                "descripcion": descripcion,
                "unidad": None,
                "cantidad": convertir_decimal_factura(
                    coincidencia.group("cantidad")
                ),
                "precio_unitario": convertir_monto_factura(
                    coincidencia.group("precio")
                ),
                "descuento": 0,
                "total_linea": convertir_monto_factura(
                    coincidencia.group("total")
                ),
                "tipo_linea": "PRODUCTO",
                "incluir_en_neto": True,
                "texto_original": limpiar_texto(coincidencia.group(0)),
            }
        )
        numero_linea += 1

    # Línea informativa que suele acompañar la factura de Transex.
    informativa = re.search(
        r"GUIAS\s+SEGUN\s+GUIAS\s*:\s*([0-9,\s]+)",
        quitar_acentos_factura(texto),
        re.IGNORECASE,
    )
    if informativa:
        detalle.append(
            {
                "nro_linea": numero_linea,
                "codigo": None,
                "descripcion": "GUIAS SEGUN GUIAS: " + limpiar_texto(informativa.group(1)),
                "unidad": None,
                "cantidad": None,
                "precio_unitario": None,
                "descuento": 0,
                "total_linea": 0,
                "tipo_linea": "INFORMATIVA",
                "incluir_en_neto": False,
                "texto_original": limpiar_texto(informativa.group(0)),
            }
        )

    return detalle
