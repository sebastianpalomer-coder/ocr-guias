import io
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageOps
import pytesseract
import pypdfium2 as pdfium


app = FastAPI()


FORMATOS_IMAGEN = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}

FORMATO_PDF = "application/pdf"


@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "service": "ocr-guias"
    }


def preparar_imagen(imagen: Image.Image) -> Image.Image:
    """
    Preparación básica para mejorar Tesseract.
    Por ahora NO hacemos corrección de perspectiva.
    """

    # Convertir a escala de grises
    imagen = imagen.convert("L")

    # Mejorar contraste automáticamente
    imagen = ImageOps.autocontrast(imagen)

    # Aumentar resolución si la imagen es pequeña
    ancho, alto = imagen.size

    if ancho < 1800:
        factor = 1800 / ancho

        imagen = imagen.resize(
            (
                int(ancho * factor),
                int(alto * factor)
            ),
            Image.Resampling.LANCZOS
        )

    return imagen


def ejecutar_ocr(imagen: Image.Image) -> str:
    """
    OCR español + inglés.
    PSM 6 funciona razonablemente bien en
    documentos estructurados.
    """

    imagen = preparar_imagen(imagen)

    config = "--oem 3 --psm 6"

    texto = pytesseract.image_to_string(
        imagen,
        lang="spa+eng",
        config=config
    )

    return texto.strip()


def procesar_pdf(datos: bytes):
    """
    Convierte cada página PDF a imagen
    y ejecuta OCR.
    """

    textos = []

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as tmp:

        tmp.write(datos)
        ruta_pdf = tmp.name

    try:

        pdf = pdfium.PdfDocument(ruta_pdf)

        total_paginas = len(pdf)

        for numero_pagina in range(total_paginas):

            pagina = pdf[numero_pagina]

            # scale 3 ≈ buena resolución para OCR
            bitmap = pagina.render(scale=3)

            imagen = bitmap.to_pil()

            texto = ejecutar_ocr(imagen)

            textos.append(
                f"--- PAGINA {numero_pagina + 1} ---\n"
                f"{texto}"
            )

        return "\n\n".join(textos), total_paginas

    finally:

        try:
            os.remove(ruta_pdf)
        except:
            pass


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

        # ------------------------------
        # PDF
        # ------------------------------

        if tipo == FORMATO_PDF:

            texto, paginas = procesar_pdf(datos)

        # ------------------------------
        # JPG / PNG
        # ------------------------------

        elif tipo in FORMATOS_IMAGEN:

            imagen = Image.open(
                io.BytesIO(datos)
            )

            texto = ejecutar_ocr(imagen)

            paginas = 1

        else:

            raise HTTPException(
                status_code=400,
                detail=f"Formato no soportado: {tipo}"
            )

        return {
            "ok": True,
            "filename": file.filename,
            "content_type": tipo,
            "pages": paginas,
            "text": texto
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
