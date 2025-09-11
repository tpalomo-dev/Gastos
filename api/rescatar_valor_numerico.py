import re

UNIDADES = {
    "cero":0, "uno":1, "una":1, "dos":2, "tres":3, "cuatro":4, "cinco":5, 
    "seis":6, "siete":7, "ocho":8, "nueve":9, "diez":10, "once":11, "doce":12, 
    "trece":13, "catorce":14, "quince":15, "dieciséis":16, "dieciseis":16, 
    "diecisiete":17, "dieciocho":18, "diecinueve":19, "veinte":20, "veintiuno":21, 
    "veintidos":22, "veintidós":22, "veintitrés":23, "veintitres":23, "veinticuatro":24, 
    "veinticinco":25, "veintiseis":26, "veintiséis":26, "veintisiete":27, 
    "veintiocho":28, "veintinueve":29
}

DECENAS = {
    "treinta":30, "cuarenta":40, "cincuenta":50, "sesenta":60, 
    "setenta":70, "ochenta":80, "noventa":90
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
    texto = re.sub(r'[^\d.,]', '', texto)
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
    
    total += parcial
    return total

def separar_texto_valor(texto):
    valores = []
    texto_limpio = texto
    
    # Encontrar números en dígitos
    numeros_digitos = re.finditer(r'[$]?\d[\d.,]*', texto)
    for match in numeros_digitos:
        numero_str = match.group()
        try:
            valor = int(normalizar_numero_digitos(numero_str))
            valores.append(valor)
        except:
            pass
    
    # Encontrar números en palabras
    todas_las_palabras = list(UNIDADES.keys()) + list(DECENAS.keys()) + list(CENTENAS.keys()) + list(MULTIPLICADORES.keys())
    patron_palabras = r'\b(?:' + '|'.join(todas_las_palabras) + r')(?:\s+(?:' + '|'.join(todas_las_palabras) + r'))*\b'
    
    matches_palabras = re.finditer(patron_palabras, texto.lower())
    for match in matches_palabras:
        secuencia = match.group()
        valor = palabras_a_numero(secuencia)
        valores.append(valor)
    
    # Limpiar texto removiendo números
    texto_limpio = re.sub(r'[$]?\d[\d.,]*', '', texto_limpio)
    texto_limpio = re.sub(patron_palabras, '', texto_limpio, flags=re.IGNORECASE)
    
    # Limpiar espacios y puntuación sobrante
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip(' ,.;')
    
    return texto_limpio, sum(valores)

# Pruebas
print(separar_texto_valor("Uber Delivery, $25,000 pesos."))
print(separar_texto_valor("Compra de cinco mil pesos"))
print(separar_texto_valor("Transferencia de $1,500 más dos mil en efectivo"))
print(separar_texto_valor("Pago de cuatrocientos cincuenta por servicios"))
print(separar_texto_valor("Compra $150 y propina de veinte pesos"))