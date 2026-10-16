"""Utilidades de la tarea de Modelos No Lineales y SVM.

Este archivo YA ESTÁ COMPLETO: no hay que modificarlo. Contiene las
operaciones sobre vectores dispersos (diccionarios `caracteristica -> valor`),
la lectura del corpus de reseñas y la representación numérica que usa el
Problema 3 (la de margen suave y kernels).

Convención de vectores dispersos: un vector se representa como un `dict` de
Python donde solo aparecen las coordenadas distintas de cero. Toda coordenada
ausente vale 0. Esto es imprescindible aquí: el vocabulario tiene decenas de
miles de palabras, pero cada reseña usa unas pocas decenas.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, TextIO, Tuple, TypeVar

import numpy as np

# Tipos que usamos en todo la tarea.
FeatureVector = Dict[str, float]   # phi(x): característica -> conteo
WeightVector = Dict[str, float]    # w: característica -> peso
Example = Tuple[str, int]          # (texto de la reseña, etiqueta +1 / -1)

T = TypeVar("T")

# Los datos viven junto a este archivo, no en el directorio de trabajo.
RUTA_DATOS = Path(__file__).resolve().parent / "datos"


def dot_product(d1: Dict, d2: Dict) -> float:
    """Producto punto de dos vectores dispersos.

    Recorre las claves del diccionario más corto y multiplica por el valor
    correspondiente del otro; las claves ausentes se tratan como 0.

    :param d1: vector disperso (dict de característica a número).
    :param d2: vector disperso (dict de característica a número).
    :return: el producto punto ``d1 . d2``.
    """
    if len(d1) < len(d2):
        return dot_product(d2, d1)
    return sum(d1.get(f, 0) * v for f, v in d2.items())


def increment(d1: Dict, escala: float, d2: Dict) -> None:
    """Implementa ``d1 += escala * d2`` para vectores dispersos.

    Modifica `d1` en el lugar (no devuelve nada). Es la forma correcta de
    actualizar el vector de pesos dentro del descenso por gradiente: evita
    recorrer todo el vocabulario en cada paso.

    :param d1: vector disperso que se modifica.
    :param escala: escalar por el que se multiplica `d2`.
    :param d2: vector disperso que se suma.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * escala


def read_examples(nombre: str) -> List[Example]:
    """Lee un archivo de reseñas y devuelve la lista de ejemplos (x, y).

    Cada línea del archivo tiene el formato ``<etiqueta> <texto>``, donde la
    etiqueta es ``+1`` (reseña positiva) o ``-1`` (reseña negativa).

    :param nombre: nombre del archivo dentro de la carpeta ``datos/``
        (por ejemplo ``"resenas.train"``), o una ruta completa.
    :return: lista de pares ``(texto, etiqueta)``.
    """
    ruta = Path(nombre)
    if not ruta.is_absolute() and not ruta.exists():
        ruta = RUTA_DATOS / nombre
    ejemplos: List[Example] = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            y, x = linea.split(" ", 1)
            ejemplos.append((x.strip(), int(y)))
    return ejemplos


def evaluate_predictor(examples: List[Tuple[T, int]],
                       predictor: Callable[[T], int]) -> float:
    """Devuelve la tasa de error de un predictor sobre una lista de ejemplos.

    :param examples: lista de pares ``(x, y)``.
    :param predictor: función que recibe un `x` y devuelve la etiqueta predicha.
    :return: fracción de ejemplos mal clasificados, entre 0 y 1.
    """
    if not examples:
        return 0.0
    errores = sum(1 for x, y in examples if predictor(x) != y)
    return errores / len(examples)


def predecir(x: T, feature_extractor: Callable[[T], FeatureVector],
             weights: WeightVector) -> int:
    """Aplica el clasificador lineal: ``+1`` si el puntaje es >= 0, si no ``-1``.

    Ojo con el empate: por convención de la tarea, un puntaje exactamente 0
    se predice como ``+1``.
    """
    return 1 if dot_product(feature_extractor(x), weights) >= 0 else -1


def output_weights(weights: WeightVector, ruta: str) -> None:
    """Escribe los pesos ordenados de mayor a menor en un archivo de texto.

    Sirve para mirar qué palabras aprendió el modelo: arriba quedan las más
    positivas y abajo las más negativas.
    """
    with open(ruta, "w", encoding="utf-8") as out:
        for f, v in sorted(weights.items(), key=lambda par: -par[1]):
            print(f"{f}\t{v}", file=out)


def verbose_predict(phi: FeatureVector, y: Optional[int],
                    weights: WeightVector, out: TextIO) -> int:
    """Imprime la predicción de una reseña descompuesta característica a característica.

    Cada línea muestra ``caracteristica  valor * peso = aporte``, ordenada por
    el aporte al puntaje. Es la herramienta central del análisis del error:
    permite ver qué palabras empujaron la decisión hacia el lado equivocado.
    """
    yy = 1 if dot_product(phi, weights) >= 0 else -1
    if y is not None:
        print("Verdad: %s, Predicción: %s [%s]"
              % (y, yy, "CORRECTA" if y == yy else "INCORRECTA"), file=out)
    else:
        print("Predicción:", yy, file=out)
    for f, v in sorted(phi.items(), key=lambda par: -par[1] * weights.get(par[0], 0)):
        w = weights.get(f, 0)
        print("%-30s%s * %s = %s" % (f, v, w, v * w), file=out)
    return yy


def output_error_analysis(examples: List[Example],
                          feature_extractor: Callable[[str], FeatureVector],
                          weights: WeightVector, ruta: str) -> None:
    """Vuelca a un archivo el desglose de la predicción de cada ejemplo."""
    with open(ruta, "w", encoding="utf-8") as out:
        for x, y in examples:
            print("===", x, file=out)
            verbose_predict(feature_extractor(x), y, weights, out)


############################################################
# Representación numérica para el Problema 3 (margen suave y kernels)
############################################################

# Las dos listas de sentimiento se construyen a partir del CONJUNTO DE
# ENTRENAMIENTO, nunca del de validación, con una regla fija y determinista:
#
#   1. se considera toda palabra que aparezca en al menos MINIMO_APARICIONES
#      reseñas de entrenamiento;
#   2. se ordena por log((positivas + 1) / (negativas + 1)), desempatando
#      alfabéticamente;
#   3. las TAMANO_LEXICO primeras forman el léxico negativo y las
#      TAMANO_LEXICO últimas, el positivo.
#
# El resultado no depende de la versión de Python ni del sistema operativo, así
# que la matriz de características es la misma para todo el curso.
MINIMO_APARICIONES = 3
TAMANO_LEXICO = 600

_LEXICOS: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None


def lexicos_de_sentimiento() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Devuelve ``(positivas, negativas)``: las dos listas de palabras.

    Se calculan una sola vez y quedan en caché. Ver el comentario de arriba
    para la regla exacta de construcción.

    :return: par de tuplas de palabras, cada una de largo ``TAMANO_LEXICO``.
    """
    global _LEXICOS
    if _LEXICOS is not None:
        return _LEXICOS

    import collections
    import math

    positivas: Dict[str, int] = collections.Counter()
    negativas: Dict[str, int] = collections.Counter()
    for texto, etiqueta in read_examples("resenas.train"):
        vistas = set(texto.split())
        destino = positivas if etiqueta == 1 else negativas
        destino.update(vistas)

    candidatas = []
    for palabra in set(positivas) | set(negativas):
        total = positivas.get(palabra, 0) + negativas.get(palabra, 0)
        if total < MINIMO_APARICIONES:
            continue
        razon = math.log((positivas.get(palabra, 0) + 1)
                         / (negativas.get(palabra, 0) + 1))
        candidatas.append((razon, palabra))
    candidatas.sort()

    negativo = tuple(p for _, p in candidatas[:TAMANO_LEXICO])
    positivo = tuple(p for _, p in candidatas[-TAMANO_LEXICO:])
    _LEXICOS = (positivo, negativo)
    return _LEXICOS


def matriz_de_resenas(examples: List[Example]) -> Tuple[np.ndarray, np.ndarray]:
    """Convierte una lista de reseñas en la matriz numérica del Problema 3.

    Cada reseña se resume en DOS números, de modo que todo el corpus cabe en
    una matriz densa de ``numpy`` y cada reseña es un punto del plano:

        x1 = cuántas palabras de la reseña están en el léxico positivo,
        x2 = cuántas palabras de la reseña están en el léxico negativo.

    Las palabras repetidas se cuentan tantas veces como aparecen.

    :param examples: lista de pares ``(texto, etiqueta)``.
    :return: ``(X, y)`` con ``X`` de forma ``(n, 2)`` e ``y`` de forma ``(n,)``,
        ambos de tipo ``float``, con las etiquetas en ``{-1, +1}``.
    """
    positivo, negativo = lexicos_de_sentimiento()
    conjunto_positivo, conjunto_negativo = set(positivo), set(negativo)

    X = np.zeros((len(examples), 2), dtype=float)
    y = np.zeros(len(examples), dtype=float)
    for i, (texto, etiqueta) in enumerate(examples):
        palabras = texto.split()
        X[i, 0] = sum(1 for p in palabras if p in conjunto_positivo)
        X[i, 1] = sum(1 for p in palabras if p in conjunto_negativo)
        y[i] = etiqueta
    return X, y


def objetivo_svm(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                 C: float) -> float:
    """Valor de la función que minimiza el SVM de margen suave.

        J(w, b) = (1/2) ||w||^2 + (C/n) sum_i max(0, 1 - y_i (w . x_i + b))

    Está aquí completa para que puedas verificar que tu entrenamiento la hace
    bajar: si no baja, el paso es demasiado grande o el signo del gradiente
    está invertido.

    :param X: matriz de características, de forma ``(n, d)``.
    :param y: etiquetas en ``{-1, +1}``, de forma ``(n,)``.
    :param w: vector de pesos, de forma ``(d,)``.
    :param b: sesgo (escalar).
    :param C: peso del término de error frente al de margen.
    :return: el valor de ``J(w, b)``.
    """
    margenes = y * (X @ np.asarray(w, dtype=float) + b)
    perdida = np.maximum(0.0, 1.0 - margenes).sum() / len(y)
    return float(0.5 * float(np.dot(w, w)) + C * perdida)


def paso_seguro(X: np.ndarray, C: float) -> float:
    """El paso más grande con el que el descenso por gradiente no oscila.

    La parte suave de ``J`` tiene curvatura acotada por
    ``1 + C * media_i ||x_i||^2``, así que el paso ``1 / (1 + C * media)``
    garantiza que cada iteración baje el objetivo. Es la fórmula que usan el
    corrector y el experimento de cierre; úsala tú también para que tus
    números coincidan con los de la pauta.

    :param X: matriz de características, de forma ``(n, d)``.
    :param C: el mismo ``C`` del objetivo.
    :return: el tamaño de paso.
    """
    escala = float((np.asarray(X, dtype=float) ** 2).sum(axis=1).mean())
    return 1.0 / (1.0 + C * escala)


def vectores_dispersos_de_matriz(X: np.ndarray) -> List[FeatureVector]:
    """Traduce cada fila de `X` al formato de diccionario del Problema 1.

    Sirve para el Problema 3c: permite entrenar con ``learn_predictor`` sobre
    exactamente las mismas características que usa el SVM, de modo que la
    comparación aísle el efecto del margen y no el de la representación.

    :param X: matriz de características, de forma ``(n, d)``.
    :return: lista de `n` diccionarios ``{"x0": ..., "x1": ...}``, sin las
        coordenadas que valen cero.
    """
    filas: List[FeatureVector] = []
    for fila in np.asarray(X, dtype=float):
        filas.append({f"x{j}": float(v) for j, v in enumerate(fila) if v != 0.0})
    return filas
