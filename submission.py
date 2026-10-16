"""Tarea: Modelos No Lineales y SVM — archivo único de entrega.

Este es el ÚNICO archivo que se entrega. Trae dos cosas:

  1. la sección de RESPUESTAS del comienzo, con la reseña del Problema 2d y
     los números que produce tu propio código en el Problema 3c, y
  2. las doce funciones marcadas con ``# BEGIN_TU_CODIGO``.

No modifiques las firmas ni los nombres (ni de las funciones ni de las
constantes): el corrector automático los busca tal cual.

Reglas:
  * Problemas 1 y 2: solo biblioteca estándar de Python. Nada de numpy,
    scikit-learn ni nltk.
  * Problema 3: numpy y nada más. Sigue prohibido scikit-learn, incluso para
    comparar: el objetivo es que el margen suave y el kernel salgan de tus
    manos.
  * Usa ``dot_product`` e ``increment`` de ``util.py`` en vez de manipular los
    diccionarios a mano: así aprovechas la representación dispersa y las
    actualizaciones no recorren todo el vocabulario.
  * Una palabra es simplemente una cadena separada por espacios en blanco.
    "Buena" y "buena" son palabras DISTINTAS (no normalizamos mayúsculas).

Para probar tu implementación completa sobre el corpus real:

    python submission.py            # entrena con características de palabras
    python submission.py ngramas    # barrido de n-gramas (Problema 2c)
    python submission.py svm        # barrido de C y comparaciones (Problema 3c)
"""

import sys
from typing import Callable, Dict, List, Tuple, TypeVar

import numpy as np

import util
from util import (Example, FeatureVector, WeightVector, dot_product,
                  evaluate_predictor, increment, output_error_analysis,
                  output_weights, read_examples)

T = TypeVar("T")


##########################################################################
##########################################################################
##                                                                      ##
##  RESPUESTAS                                                          ##
##                                                                      ##
##  Acá va lo que no es código: la reseña que construyes en el          ##
##  Problema 2d y los números que produce tu propio programa en el      ##
##  Problema 3c. Esta sección se completa al final, cuando las doce     ##
##  funciones ya están listas y los dos comandos de más abajo corren.   ##
##                                                                      ##
##  Reemplaza cada valor por el tuyo, sin cambiar los nombres de las    ##
##  constantes ni agregar texto adicional dentro de ellas.              ##
##                                                                      ##
##  Cómo se comparan las respuestas:                                    ##
##                                                                      ##
##  * los números se comparan con TRES decimales, así que da lo mismo   ##
##    anotar 0,1234 o 0,123;                                            ##
##  * las claves de cada diccionario tienen que ser exactamente las     ##
##    que dice el comentario, ni una más ni una menos.                  ##
##                                                                      ##
##  `grader.py` no revisa esta sección: sus puntos los pone el          ##
##  corrector del curso. Debajo empieza el código, que sí revisan las   ##
##  pruebas de `grader.py`.                                             ##
##                                                                      ##
##########################################################################
##########################################################################


############################################################
# Problema 2d: una reseña donde los n-gramas le ganan a las palabras
############################################################

# Una reseña NEGATIVA, de al menos cuatro palabras, tal que:
#   * ninguna de sus palabras aparezca en el vocabulario de palabras del
#     conjunto de entrenamiento, y
#   * el clasificador de 6-gramas de caracteres (20 épocas, eta = 0,01,
#     entrenado sobre el corpus real) le dé puntaje negativo.
# BEGIN_TU_CODIGO
P2_RESENA = ''
# END_TU_CODIGO


############################################################
# Problema 3c: el barrido de C
#
# Los números salen de la primera tabla que imprime
#
#     python submission.py svm
#
# Las claves son los cuatro valores de C del barrido, tal cual: 0.1, 1,
# 10 y 100.
############################################################

# Cuántos vectores de soporte quedan con cada C (la columna «vec. sop.»,
# el conteo y no el porcentaje).
# BEGIN_TU_CODIGO
SVM_VECTORES_DE_SOPORTE = {0.1: None, 1: None, 10: None, 100: None}
# END_TU_CODIGO

# Error de entrenamiento de cada uno de esos clasificadores.
# BEGIN_TU_CODIGO
SVM_ERROR_ENTRENAMIENTO = {0.1: None, 1: None, 10: None, 100: None}
# END_TU_CODIGO

# Error de validación de cada uno de esos clasificadores.
# BEGIN_TU_CODIGO
SVM_ERROR_VALIDACION = {0.1: None, 1: None, 10: None, 100: None}
# END_TU_CODIGO


############################################################
# Problema 3c: la frontera curva y el SGD sin regularización
#
# Los números salen de los otros dos bloques del mismo comando, sobre
# exactamente las mismas dos características.
############################################################

# Error de validación del SVM entrenado sobre el mapa polinomial de grado 2,
# con las mismas cuatro claves del barrido de C.
# BEGIN_TU_CODIGO
SVM_GRADO2_ERROR_VALIDACION = {0.1: None, 1: None, 10: None, 100: None}
# END_TU_CODIGO

# Error de validación del SGD sin regularización del Problema 1b entrenado
# sobre esas mismas dos características (la última línea del comando).
# BEGIN_TU_CODIGO
SGD_ERROR_VALIDACION = None
# END_TU_CODIGO


##########################################################################
##########################################################################
##                                                                      ##
##  CÓDIGO                                                              ##
##                                                                      ##
##  De acá en adelante van las doce funciones que debes programar.      ##
##                                                                      ##
##########################################################################
##########################################################################


############################################################
# Problema 1a: extracción de características de palabras
############################################################


def extract_word_features(x: str) -> FeatureVector:
    """Extrae las características de palabras de la reseña `x`.

    Las palabras se separan únicamente por espacios en blanco, y la
    característica asociada a cada palabra es la cantidad de veces que
    aparece en la reseña.

    Ejemplo:
        "me gusta y me gusta" --> {'me': 2, 'gusta': 2, 'y': 1}

    :param x: texto de la reseña.
    :return: vector disperso ``phi(x)`` como diccionario.
    """
    # BEGIN_TU_CODIGO
    counts: FeatureVector = {}
    for word in x.split():
        counts[word] = counts.get(word, 0) + 1
    return counts
    # END_TU_CODIGO


############################################################
# Problema 1b: descenso por gradiente estocástico
############################################################


def learn_predictor(train_examples: List[Tuple[T, int]],
                    validation_examples: List[Tuple[T, int]],
                    feature_extractor: Callable[[T], FeatureVector],
                    num_epochs: int,
                    eta: float,
                    verbose: bool = True) -> WeightVector:
    """Entrena un clasificador lineal con SGD sobre la pérdida hinge.

    En cada paso se toma UN ejemplo ``(x, y)``, se calcula el margen
    ``m = (w . phi(x)) * y`` y se actualiza

        w <- w - eta * grad_w Perd_hinge(x, y, w),

    con el gradiente de la pérdida hinge.

    :param train_examples: lista de pares ``(x, y)`` para entrenar.
    :param validation_examples: lista de pares ``(x, y)`` solo para medir
        (nunca para entrenar).
    :param feature_extractor: función que transforma `x` en ``phi(x)``.
    :param num_epochs: cantidad de pasadas completas sobre el conjunto de
        entrenamiento.
    :param eta: tamaño de paso.
    :param verbose: si es ``True``, imprime el error de entrenamiento y de
        validación al final de cada época.
    :return: el vector de pesos aprendido, como vector disperso.
    """
    weights: WeightVector = {}

    # BEGIN_TU_CODIGO
    train_data = [(feature_extractor(x), y) for x, y in train_examples]
    for epoch in range(num_epochs):
        for phi, y in train_data:
            if dot_product(weights, phi) * y < 1:
                increment(weights, eta * y, phi)
        if verbose:
            error_train = evaluate_predictor(
                train_examples, lambda x: util.predecir(x, feature_extractor, weights))
            error_val = evaluate_predictor(
                validation_examples, lambda x: util.predecir(x, feature_extractor, weights))
            print("Época %2d/%d: error entrenamiento = %.4f, error validación = %.4f"
                  % (epoch + 1, num_epochs, error_train, error_val))
    # END_TU_CODIGO

    return weights


############################################################
# Problema 1c: características de n-gramas de caracteres
############################################################


def extract_character_features(n: int) -> Callable[[str], FeatureVector]:
    """Devuelve un extractor de n-gramas de caracteres.

    La función devuelta recibe una cadena `x`, le quita los espacios en
    blanco (espacios y tabuladores) y cuenta cuántas veces aparece cada
    subcadena de largo `n`.

    Ejemplo (n = 3):
        "la vi" --> "lavi" --> {'lav': 1, 'avi': 1}

    Puedes suponer que ``1 <= n <= len(x)``.

    :param n: largo de los n-gramas.
    :return: función que va de una cadena a su vector disperso.
    """

    def extract(x: str) -> FeatureVector:
        # BEGIN_TU_CODIGO
        s = "".join(x.split())
        counts: FeatureVector = {}
        for i in range(len(s) - n + 1):
            ngram = s[i:i + n]
            counts[ngram] = counts.get(ngram, 0) + 1
        return counts
        # END_TU_CODIGO

    return extract


############################################################
# Problema 2a: análisis del error
############################################################


def error_analysis(examples: List[Example],
                   feature_extractor: Callable[[str], FeatureVector],
                   weights: WeightVector,
                   k: int = 5) -> List[Tuple[str, int, int, float]]:
    """Devuelve las `k` reseñas peor clasificadas por el modelo.

    Se consideran solo los ejemplos MAL clasificados, es decir aquellos en
    que la predicción (``+1`` si el puntaje es >= 0, si no ``-1``) no coincide
    con la etiqueta verdadera. Cada uno se describe con su margen

        margen = (w . phi(x)) * y,

    que es tanto más negativo cuanto más seguro estaba el modelo de su error.
    La lista se devuelve ordenada de margen menor a mayor (los errores más
    groseros primero) y truncada a los primeros `k` elementos. Ante empates se
    conserva el orden en que aparecen en `examples`.

    :param examples: lista de pares ``(texto, etiqueta)``.
    :param feature_extractor: función que transforma el texto en ``phi(x)``.
    :param weights: vector de pesos del clasificador.
    :param k: cantidad máxima de reseñas a devolver.
    :return: lista de tuplas ``(texto, etiqueta, predicción, margen)``.
    """
    # BEGIN_TU_CODIGO
    mal_clasificados = []
    for x, y in examples:
        phi = feature_extractor(x)
        score = dot_product(weights, phi)
        pred = 1 if score >= 0 else -1
        if pred != y:
            margen = float(score * y)
            mal_clasificados.append((x, y, pred, margen))
    mal_clasificados.sort(key=lambda t: t[3])
    return mal_clasificados[:k]
    # END_TU_CODIGO


############################################################
# Problema 2b: qué características empujaron hacia el lado equivocado
############################################################


def caracteristicas_culpables(x: str, y: int,
                              feature_extractor: Callable[[str], FeatureVector],
                              weights: WeightVector,
                              k: int = 3) -> List[Tuple[str, float]]:
    """Devuelve las `k` características de `x` que más dañaron la predicción.

    Cada característica ``f`` con valor ``v = phi(x)[f]`` y peso ``w[f]``
    aporta al margen la cantidad

        aporte(f) = v * w[f] * y.

    Una característica es CULPABLE cuando su aporte es estrictamente negativo:
    empujó el puntaje hacia el lado contrario a la etiqueta verdadera `y`.
    Devuelve solo las culpables, como tuplas ``(característica, aporte)``,
    ordenadas de aporte menor a mayor (la más dañina primero) y truncadas a
    `k`. Ante empates de aporte, ordena las características alfabéticamente.
    Si `x` no tiene ninguna característica culpable, devuelve la lista vacía.

    Ejemplo, con w = {'buena': 1.0, 'mala': -1.0, 'peli': 0.2}:
        caracteristicas_culpables("mala peli", 1, extract_word_features, w, 2)
        --> [('mala', -1.0)]

    :param x: texto de la reseña.
    :param y: etiqueta verdadera (+1 o -1).
    :param feature_extractor: función que transforma el texto en ``phi(x)``.
    :param weights: vector de pesos del clasificador.
    :param k: cantidad máxima de características a devolver.
    :return: lista de tuplas ``(característica, aporte)``.
    """
    # BEGIN_TU_CODIGO
    phi = feature_extractor(x)
    culpables = []
    for f, v in phi.items():
        w = weights.get(f, 0.0)
        aporte = float(v * w * y)
        if aporte < 0:
            culpables.append((f, aporte))
    culpables.sort(key=lambda t: (t[1], t[0]))
    return culpables[:k]
    # END_TU_CODIGO


############################################################
# Problema 2c: qué largo de n-grama conviene
############################################################


def mejor_n(train_examples: List[Example],
            validation_examples: List[Example],
            valores_de_n: List[int],
            num_epochs: int = 20,
            eta: float = 0.01) -> Tuple[int, Dict[int, float]]:
    """Compara extractores de n-gramas de caracteres y elige el mejor `n`.

    Para cada `n` de `valores_de_n`: entrena un clasificador con
    ``extract_character_features(n)`` sobre `train_examples` (con
    ``learn_predictor``, ``num_epochs`` épocas y paso `eta`) y mide su tasa de
    error sobre `validation_examples`.

    :param train_examples: lista de pares ``(texto, etiqueta)`` de entrenamiento.
    :param validation_examples: lista de pares ``(texto, etiqueta)`` de validación.
    :param valores_de_n: largos de n-grama a comparar.
    :param num_epochs: épocas de cada entrenamiento.
    :param eta: tamaño de paso de cada entrenamiento.
    :return: la tupla ``(mejor, errores)``, donde `errores` es el diccionario
        ``n -> error de validación`` con una entrada por cada `n` pedido y
        `mejor` es el `n` de menor error de validación. Si hay empate, gana el
        `n` más pequeño.
    """
    # BEGIN_TU_CODIGO
    errores: Dict[int, float] = {}
    for n in valores_de_n:
        fe = extract_character_features(n)
        weights = learn_predictor(train_examples, validation_examples, fe,
                                  num_epochs=num_epochs, eta=eta, verbose=False)
        err = evaluate_predictor(validation_examples,
                                 lambda x: util.predecir(x, fe, weights))
        errores[n] = err
    mejor = min(valores_de_n, key=lambda n: (errores[n], n))
    return mejor, errores
    # END_TU_CODIGO


############################################################
# Problema 3a: margen suave (SVM lineal)
############################################################


def entrenar_svm_margen_suave(X: np.ndarray, y: np.ndarray, C: float,
                              epocas: int, eta: float) -> Tuple[np.ndarray, float]:
    """Entrena un SVM lineal de margen suave por descenso por subgradiente.

    Minimiza, sobre TODO el conjunto de entrenamiento a la vez (no ejemplo a
    ejemplo como en el Problema 1b),

        J(w, b) = (1/2)||w||^2 + (C/n) sum_i max(0, 1 - y_i (w . x_i + b)).

    Sea ``v_i = 1`` si el ejemplo `i` viola el margen (es decir, si
    ``y_i (w . x_i + b) < 1``) y ``v_i = 0`` si no. El subgradiente es

        dJ/dw = w - (C/n) sum_i v_i y_i x_i,
        dJ/db =   - (C/n) sum_i v_i y_i,

    y cada iteración hace ``w <- w - eta * dJ/dw`` y ``b <- b - eta * dJ/db``.
    Parte de ``w = 0`` y ``b = 0`` y ejecuta exactamente `epocas` iteraciones,
    sin criterio de parada anticipada: así el resultado es reproducible.

    Escríbelo vectorizado, sin bucles sobre los ejemplos: ``X @ w`` calcula los
    `n` puntajes de una vez y ``X.T @ (y * v)`` la suma del subgradiente.

    :param X: matriz de características, de forma ``(n, d)``.
    :param y: etiquetas en ``{-1, +1}``, de forma ``(n,)``.
    :param C: peso del término de error frente al de margen.
    :param epocas: cantidad de iteraciones de descenso.
    :param eta: tamaño de paso (usa ``util.paso_seguro(X, C)``).
    :return: el par ``(w, b)`` con ``w`` de forma ``(d,)`` y ``b`` escalar.
    """
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO

    return w, b


def indices_vectores_de_soporte(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                                b: float, tol: float = 1e-6) -> np.ndarray:
    """Índices de los vectores de soporte del clasificador ``(w, b)``.

    En el margen suave son vectores de soporte los ejemplos que quedan sobre
    el margen o adentro de él, es decir aquellos con

        y_i (w . x_i + b) <= 1 + tol.

    La tolerancia evita que un ejemplo exactamente sobre el margen se pierda
    por el error de redondeo del punto flotante.

    :param X: matriz de características, de forma ``(n, d)``.
    :param y: etiquetas en ``{-1, +1}``, de forma ``(n,)``.
    :param w: vector de pesos, de forma ``(d,)``.
    :param b: sesgo (escalar).
    :param tol: holgura numérica sobre el margen.
    :return: arreglo de índices en orden creciente, de tipo entero.
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


def error_de_clasificacion(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                           b: float) -> float:
    """Fracción de ejemplos mal clasificados por ``(w, b)``.

    Se mantiene la convención de la tarea: un puntaje exactamente ``0`` se
    predice como ``+1``.

    :param X: matriz de características, de forma ``(n, d)``.
    :param y: etiquetas en ``{-1, +1}``, de forma ``(n,)``.
    :param w: vector de pesos, de forma ``(d,)``.
    :param b: sesgo (escalar).
    :return: número entre 0 y 1.
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


############################################################
# Problema 3b: kernels
############################################################


def kernel_polinomial(a: np.ndarray, b: np.ndarray, grado: int = 2,
                      coef0: float = 1.0) -> float:
    """Kernel polinomial ``K(a, b) = (a . b + coef0)^grado``.

    :param a: vector de forma ``(d,)``.
    :param b: vector de forma ``(d,)``.
    :param grado: exponente del kernel.
    :param coef0: término constante que se suma antes de elevar.
    :return: el valor del kernel, como ``float`` de Python.
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


def kernel_rbf(a: np.ndarray, b: np.ndarray, gamma: float = 0.5) -> float:
    """Kernel gaussiano ``K(a, b) = exp(-gamma * ||a - b||^2)``.

    :param a: vector de forma ``(d,)``.
    :param b: vector de forma ``(d,)``.
    :param gamma: ancho inverso del kernel; a mayor `gamma`, más angosto.
    :return: el valor del kernel, como ``float`` de Python.
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


def mapa_polinomial_grado2(x: np.ndarray) -> np.ndarray:
    """Mapa explícito del kernel polinomial de grado 2 con ``coef0 = 1``.

    Devuelve el vector ``psi(x)`` que cumple, para todo par de vectores,

        psi(a) . psi(b) = (a . b + 1)^2 = kernel_polinomial(a, b, 2, 1.0).

    Para ``x`` en R^d el vector resultante tiene ``(d+1)(d+2)/2``
    coordenadas, en este orden:

        1,
        sqrt(2) x_i                 para i = 0, ..., d-1,
        x_i^2                       para i = 0, ..., d-1,
        sqrt(2) x_i x_j             para i < j.

    El factor ``sqrt(2)`` es lo que hace que los términos cruzados aparezcan
    dos veces al desarrollar el cuadrado, igual que en el kernel.

    :param x: vector de forma ``(d,)``.
    :return: vector ``psi(x)`` de forma ``((d+1)(d+2)/2,)``.
    """
    x = np.asarray(x, dtype=float)

    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


def matriz_mapa_polinomial_grado2(X: np.ndarray) -> np.ndarray:
    """Aplica `mapa_polinomial_grado2` a cada fila de `X`.

    Ya está escrita: sirve para entrenar el SVM lineal en el espacio
    transformado y obtener así una frontera curva en el espacio original.

    :param X: matriz de forma ``(n, d)``.
    :return: matriz de forma ``(n, (d+1)(d+2)/2)``.
    """
    return np.array([mapa_polinomial_grado2(fila) for fila in X], dtype=float)


############################################################
# Programa principal (no se corrige: sirve para explorar)
############################################################


VALORES_DE_C = (0.1, 1.0, 10.0, 100.0)
EPOCAS_SVM = 2000


def barrido_de_C(epocas: int = EPOCAS_SVM) -> None:
    """Entrena el SVM de margen suave para varios `C` e informa la tabla.

    Es la tabla que pide el Problema 3c: para cada `C`, cuántos vectores de
    soporte quedan y qué error de entrenamiento y de validación se obtiene.
    Debajo imprime las otras dos cosas que van a la sección de respuestas: el
    mismo barrido sobre el mapa polinomial de grado 2 y el SGD sin
    regularización sobre esas mismas dos características. La forma cómoda de
    correrlo es

        python submission.py svm
    """
    entrenamiento = read_examples("resenas.train")
    validacion = read_examples("resenas.dev")
    X_train, y_train = util.matriz_de_resenas(entrenamiento)
    X_val, y_val = util.matriz_de_resenas(validacion)
    n = len(y_train)

    print("SVM lineal de margen suave sobre las dos características de léxico")
    print("  n = %d ejemplos de entrenamiento, d = %d características"
          % (n, X_train.shape[1]))
    print()
    print("  %8s %10s %8s %12s %12s %10s"
          % ("C", "J(w,b)", "|w|", "vec. sop.", "err. entren.", "err. val."))
    for C in VALORES_DE_C:
        eta = util.paso_seguro(X_train, C)
        w, b = entrenar_svm_margen_suave(X_train, y_train, C, epocas, eta)
        soporte = indices_vectores_de_soporte(X_train, y_train, w, b)
        print("  %8g %10.4f %8.4f %6d (%4.1f%%) %12.4f %10.4f"
              % (C, util.objetivo_svm(X_train, y_train, w, b, C),
                 float(np.linalg.norm(w)), len(soporte), 100.0 * len(soporte) / n,
                 error_de_clasificacion(X_train, y_train, w, b),
                 error_de_clasificacion(X_val, y_val, w, b)))

    print()
    print("Kernel polinomial de grado 2: SVM lineal sobre el mapa explícito")
    Z_train = matriz_mapa_polinomial_grado2(X_train)
    Z_val = matriz_mapa_polinomial_grado2(X_val)
    for C in VALORES_DE_C:
        eta = util.paso_seguro(Z_train, C)
        w, b = entrenar_svm_margen_suave(Z_train, y_train, C, epocas, eta)
        soporte = indices_vectores_de_soporte(Z_train, y_train, w, b)
        print("  C = %-6g vec. sop. = %5d  err. entren. = %.4f  err. val. = %.4f"
              % (C, len(soporte),
                 error_de_clasificacion(Z_train, y_train, w, b),
                 error_de_clasificacion(Z_val, y_val, w, b)))

    print()
    print("SGD sin regularización (Problema 1b) sobre las MISMAS dos características")
    dispersos_train = list(zip(util.vectores_dispersos_de_matriz(X_train),
                               [int(v) for v in y_train]))
    dispersos_val = list(zip(util.vectores_dispersos_de_matriz(X_val),
                             [int(v) for v in y_val]))
    identidad = lambda phi: phi  # noqa: E731 - las filas ya son phi(x)
    pesos = learn_predictor(dispersos_train, dispersos_val, identidad,
                            num_epochs=20, eta=0.01, verbose=False)
    error_train = evaluate_predictor(
        dispersos_train, lambda phi: util.predecir(phi, identidad, pesos))
    error_val = evaluate_predictor(
        dispersos_val, lambda phi: util.predecir(phi, identidad, pesos))
    norma = np.sqrt(sum(v * v for v in pesos.values()))
    print("  |w| = %.4f  err. entren. = %.4f  err. val. = %.4f"
          % (norma, error_train, error_val))


def main() -> None:
    """Entrena con características de palabras y deja los archivos de análisis.

    Genera dos archivos en el directorio actual:
      * ``pesos.txt``: las características ordenadas por peso.
      * ``analisis-error.txt``: el desglose de la predicción de cada reseña
        del conjunto de validación.
    """
    train_examples = read_examples("resenas.train")
    validation_examples = read_examples("resenas.dev")
    print("Leídos %d ejemplos de entrenamiento y %d de validación."
          % (len(train_examples), len(validation_examples)))

    weights = learn_predictor(train_examples, validation_examples,
                              extract_word_features, num_epochs=20, eta=0.01)
    output_weights(weights, "pesos.txt")
    output_error_analysis(validation_examples, extract_word_features, weights,
                          "analisis-error.txt")

    print("\nLas 5 reseñas peor clasificadas del conjunto de validación:")
    for texto, y, prediccion, margen in error_analysis(
            validation_examples, extract_word_features, weights, k=5):
        print("  margen = %+.3f | verdad = %+d | predicción = %+d | %s"
              % (margen, y, prediccion, texto[:90]))
        for f, aporte in caracteristicas_culpables(
                texto, y, extract_word_features, weights, k=3):
            print("      culpable: %-20s aporte = %+.3f" % (f, aporte))


def barrido_de_ngramas() -> None:
    """Corre `mejor_n` sobre el corpus real para n = 2, ..., 7 (Problema 2c)."""
    train_examples = read_examples("resenas.train")
    validation_examples = read_examples("resenas.dev")
    print("Barrido de n-gramas de caracteres (esto entrena un modelo por n)...")
    mejor, errores = mejor_n(train_examples, validation_examples,
                             list(range(2, 8)))
    for n in sorted(errores):
        print("  n = %d -> error de validación = %.4f" % (n, errores[n]))
    print("  el mejor n del barrido es %d" % mejor)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ngramas":
        barrido_de_ngramas()
    elif len(sys.argv) > 1 and sys.argv[1] == "svm":
        barrido_de_C()
    elif len(sys.argv) > 1:
        print("Uso: python submission.py [ngramas | svm]")
        sys.exit(1)
    else:
        main()
