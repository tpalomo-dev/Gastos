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
    texto = re.sub(r'[^\d]', '', texto)  # solo dígitos
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
    total += parcial
    return total

def separar_texto_valor(texto):
    total = 0
    
    # Extraer números en dígitos
    numeros_digitos = re.findall(r'\d[\d.,]*', texto)
    for n in numeros_digitos:
        total += int(normalizar_numero_digitos(n))
    
    # Extraer números en palabras
    total += palabras_a_numero(texto)
    
    # Limpiar texto de palabras numéricas y dígitos
    todas_las_palabras_numeros = set(UNIDADES.keys()) | set(DECENAS.keys()) | set(CENTENAS.keys()) | set(MULTIPLICADORES.keys())
    tokens = re.split(r'(\s+)', texto)  # mantenemos espacios
    texto_final = ''.join([
        t for t in tokens
        if t.strip().lower() not in todas_las_palabras_numeros and not re.match(r'^\d[\d.,]*$', t.strip())
    ])
    
    texto_final = texto_final.strip(",. $")  # quitamos signos sobrantes
    
    return texto_final, total

print(separar_texto_valor("Uber Delivery, $25,000 pesos."))
