import re

UNIDADES = {
    "cero":0, "uno":1, "una":1, "dos":2, "tres":3, "cuatro":4, "cinco":5,
    "seis":6, "siete":7, "ocho":8, "nueve":9, "diez":10, "once":11,
    "doce":12, "trece":13, "catorce":14, "quince":15, "dieciséis":16,
    "dieciseis":16, "diecisiete":17, "dieciocho":18, "diecinueve":19,
    "veinte":20, "veintiuno":21, "veintidos":22, "veintidós":22, "veintitrés":23,
    "veintitres":23, "veinticuatro":24, "veinticinco":25, "veintiseis":26,
    "veintiséis":26, "veintisiete":27, "veintiocho":28, "veintinueve":29
}

DECENAS = {
    "treinta":30, "cuarenta":40, "cincuenta":50, "sesenta":60, "setenta":70,
    "ochenta":80, "noventa":90
}

CENTENAS = {
    "cien":100, "ciento":100, "doscientos":200, "trescientos":300,
    "cuatrocientos":400, "quinientos":500, "seiscientos":600,
    "setecientos":700, "ochocientos":800, "novecientos":900
}

MULTIPLICADORES = {
    "mil":1000, "millón":1000000, "millones":1000000
}

def normalizar_numero_digitos(texto):
    # Eliminar todo lo que no sea dígito, punto o coma
    texto = re.sub(r'[^\d.,]', '', texto)
    # Reemplazamos separadores de miles por nada (ej: 25,000 -> 25000)
    texto = re.sub(r'(?<=\d)[.,](?=\d{3}\b)', '', texto)
    return texto

def palabras_a_numero(texto):
    texto = texto.lower().replace(" y ", " ")
    tokens = re.split(r'[\s-]+', texto)
    total = 0
    parcial = 0
    for t in tokens:
        if t in UNIDADES:
            parcial += UNIDADES[t]
        elif t in DECENAS:
            parcial += DECENAS[t]
        elif t in CENTENAS:
            parcial += CENTENAS[t]
        elif t in MULTIPLICADORES:
            mult = MULTIPLICADORES[t]
            if parcial == 0:
                parcial = 1
            parcial *= mult
            if mult >= 1000:
                total += parcial
                parcial = 0
        else:
            # intentamos convertir números ya normalizados
            try:
                parcial += int(normalizar_numero_digitos(t))
            except:
                pass
    total += parcial
    return total

def separar_texto_valor(texto):
    # Normalizamos los números en dígitos
    texto_normalizado = normalizar_numero_digitos(texto)
    
    # Calculamos el valor total
    total = palabras_a_numero(texto_normalizado)
    
    # Limpiamos el texto de palabras numéricas
    todas_las_palabras_numeros = set(UNIDADES.keys()) | set(DECENAS.keys()) | set(CENTENAS.keys()) | set(MULTIPLICADORES.keys())
    
    texto_final = ' '.join([
    w for w in re.split(r'[\s-]+', texto)
    if w.lower() not in todas_las_palabras_numeros 
    and not re.search(r'\d', w)  # elimina cualquier token que tenga un dígito
    ])
    
    return texto_final.strip(), total

print(separar_texto_valor("Uber delivery $25,000"))
