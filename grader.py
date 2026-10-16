#!/usr/bin/env python3
"""Corrector automático de la tarea de Modelos No Lineales y SVM.

Se ejecuta sin red y sin más dependencia que numpy:

    python grader.py            # corre todas las pruebas de esta copia
    python grader.py 1b-2       # corre solo una prueba

Acá están las pruebas visibles: revisan el CÓDIGO de ``submission.py`` y suman
60 de los 100 puntos de la tarea. Hay al menos una por cada función que se
programa y, cuando fallan, el mensaje dice exactamente qué se esperaba y qué se
obtuvo, así que sirven para depurar.

Los 40 puntos que faltan los pone el corrector del curso, que corre estas
mismas pruebas y agrega otras: más casos sobre las mismas funciones (escala,
bordes y consistencia) y la revisión de la sección de respuestas de
``submission.py`` (la reseña del Problema 2d y los números del Problema 3c).
Ese puntaje también se reparte solo. Llegar a 60/60 acá es condición necesaria
para el puntaje completo, no suficiente.

El código de salida es 0 solo si se obtiene el puntaje completo de esta copia.
"""

import contextlib
import re
import signal
import sys

# Nada de .pyc en la carpeta de una entrega: el bytecode de submission
# es la solución de referencia en forma decompilable.
sys.dont_write_bytecode = True
import traceback
from typing import Callable, List, Optional

import numpy as np

import util
from util import evaluate_predictor, read_examples

TOLERANCIA = 1e-6

# Tolerancia relativa de las comprobaciones numéricas del Problema 3. Se
# aplica como |obtenido - esperado| <= TOLERANCIA_RELATIVA * max(1, |esperado|),
# de modo que sirve tanto para valores del orden de 10^2 como de 10^-6.
TOLERANCIA_RELATIVA = 1e-9


############################################################
# Motor mínimo de corrección
############################################################


class FalloDePrueba(Exception):
    """Se lanza cuando una comprobación de una prueba no se cumple."""


class TiempoAgotado(Exception):
    """Se lanza cuando una prueba supera su límite de tiempo."""


def _alarma(signum, frame):  # pragma: no cover - manejador de señal
    raise TiempoAgotado()


@contextlib.contextmanager
def limite_de_tiempo(segundos: int):
    """Corta la prueba si se demora más de `segundos` (donde el SO lo permita).

    Windows no implementa ``signal.SIGALRM`` ni ``signal.alarm``: ahí el
    temporizador se desactiva y las pruebas corren sin límite de tiempo, en vez
    de hacer caer el corrector completo.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    anterior = signal.signal(signal.SIGALRM, _alarma)
    signal.alarm(segundos + 1)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, anterior)


def clave_de_orden(numero: str):
    """Clave de orden natural para los códigos de parte («1b-2», «3a-1»).

    Separa dígitos de no dígitos para que «1b-2» quede antes que «1b-10» y para
    que las partes registradas después de la carga del módulo se intercalen en
    el lugar que les corresponde, en vez de amontonarse al final del informe.
    """
    return [(0, int(t)) if t.isdigit() else (1, t)
            for t in re.findall(r"\d+|\D+", numero)]


class Parte:
    def __init__(self, numero: str, descripcion: str, puntos: float,
                 funcion: Callable, segundos: int = 5,
                 oculta: bool = False):
        self.numero = numero
        self.descripcion = descripcion
        self.puntos_max = puntos
        self.funcion = funcion
        self.segundos = segundos
        self.oculta = oculta
        self.puntos = 0.0


class Corrector:
    def __init__(self) -> None:
        self.partes: List[Parte] = []
        # Rótulo de la última fila del informe. La copia que corre por sí sola
        # reparte un subtotal, no el puntaje de la tarea.
        self.etiqueta_total = "SUBTOTAL"

    # -------- registro de partes --------

    def automatica(self, numero: str, funcion: Callable, puntos: float,
                   descripcion: str, segundos: int = 5,
                   oculta: bool = False) -> None:
        self.partes.append(Parte(numero, descripcion, puntos, funcion,
                                 segundos, oculta))

    # -------- comprobaciones --------

    @staticmethod
    def _iguales(esperado, obtenido) -> bool:
        if isinstance(esperado, float) or isinstance(obtenido, float):
            try:
                return abs(float(esperado) - float(obtenido)) < TOLERANCIA
            except (TypeError, ValueError):
                return False
        if isinstance(esperado, dict) and isinstance(obtenido, dict):
            if set(esperado) != set(obtenido):
                return False
            return all(Corrector._iguales(v, obtenido[k])
                       for k, v in esperado.items())
        if (isinstance(esperado, (list, tuple))
                and isinstance(obtenido, (list, tuple))):
            if len(esperado) != len(obtenido):
                return False
            return all(Corrector._iguales(a, b)
                       for a, b in zip(esperado, obtenido))
        return esperado == obtenido

    def es_igual(self, esperado, obtenido, contexto: str = "") -> None:
        if not self._iguales(esperado, obtenido):
            self.falla("se esperaba %s y se obtuvo %s%s"
                       % (self._corto(esperado), self._corto(obtenido),
                          f" ({contexto})" if contexto else ""))

    def es_cercano(self, esperado: float, obtenido, rel: float = TOLERANCIA_RELATIVA,
                   contexto: str = "") -> None:
        """Compara dos números con tolerancia RELATIVA.

        La comprobación es ``|obtenido - esperado| <= rel * max(1, |esperado|)``,
        que sirve igual para valores grandes (un kernel polinomial puede valer
        miles) que para valores diminutos (un kernel gaussiano lejano vale
        10^-6). La tolerancia absoluta de `es_igual` no serviría para los
        segundos: aceptaría cero como respuesta.
        """
        try:
            valor = float(obtenido)
        except (TypeError, ValueError):
            self.falla("se esperaba un número y se obtuvo %s%s"
                       % (self._corto(obtenido), f" ({contexto})" if contexto else ""))
            return
        if not abs(valor - esperado) <= rel * max(1.0, abs(esperado)):
            self.falla("se esperaba %.12g y se obtuvo %.12g%s"
                       % (esperado, valor, f" ({contexto})" if contexto else ""))

    def es_menor_que(self, cota: float, valor: float, contexto: str = "") -> None:
        if not valor < cota:
            self.falla("se esperaba un valor menor que %s y se obtuvo %s%s"
                       % (cota, valor, f" ({contexto})" if contexto else ""))

    def es_mayor_que(self, cota: float, valor: float, contexto: str = "") -> None:
        if not valor > cota:
            self.falla("se esperaba un valor mayor que %s y se obtuvo %s%s"
                       % (cota, valor, f" ({contexto})" if contexto else ""))

    def es_verdadero(self, condicion: bool, contexto: str = "") -> None:
        if not condicion:
            self.falla("la condición no se cumple%s"
                       % (f" ({contexto})" if contexto else ""))

    @staticmethod
    def falla(mensaje: str) -> None:
        raise FalloDePrueba(mensaje)

    @staticmethod
    def _corto(valor, largo: int = 220) -> str:
        texto = str(valor)
        return texto if len(texto) <= largo else texto[:largo] + "..."

    # -------- ejecución --------

    def _correr_parte(self, parte: Parte) -> None:
        etiqueta = "OCULTA " if parte.oculta else "visible"
        print(f"--- Parte {parte.numero} [{etiqueta}] {parte.descripcion}")
        try:
            with limite_de_tiempo(parte.segundos):
                parte.funcion()
            parte.puntos = parte.puntos_max
            print(f"    OK  {parte.puntos:g}/{parte.puntos_max:g} puntos")
        except FalloDePrueba as e:
            detalle = (str(e) if not parte.oculta
                       else "la prueba oculta no entrega detalles")
            print(f"    FALLA: {detalle}")
            print(f"    0/{parte.puntos_max:g} puntos")
        except TiempoAgotado:
            print(f"    FALLA: se superó el límite de {parte.segundos} s")
            print(f"    0/{parte.puntos_max:g} puntos")
        except NotImplementedError as e:
            print(f"    FALLA: función sin implementar ({e})")
            print(f"    0/{parte.puntos_max:g} puntos")
        except Exception as e:  # noqa: BLE001 - queremos reportar cualquier error
            print(f"    FALLA: excepción {type(e).__name__}: {e}")
            for linea in traceback.format_exc().splitlines()[-4:]:
                print(f"      {linea}")
            print(f"    0/{parte.puntos_max:g} puntos")
        print()

    def correr(self, seleccion: Optional[str] = None) -> int:
        partes = sorted((p for p in self.partes
                         if seleccion is None or p.numero == seleccion),
                        key=lambda p: clave_de_orden(p.numero))
        if not partes:
            print(f"No existe la parte «{seleccion}».")
            return 1

        print("=" * 68)
        print("Tarea: Modelos No Lineales y SVM — corrección automática")
        print("=" * 68)
        print()

        for parte in partes:
            self._correr_parte(parte)

        obtenido = sum(p.puntos for p in partes)
        maximo = sum(p.puntos_max for p in partes)

        print("=" * 68)
        print("PUNTAJE POR PARTE")
        for p in partes:
            print("  %-8s %-52s %4.1f/%.1f"
                  % (p.numero, p.descripcion[:52], p.puntos, p.puntos_max))
        print("  %-8s %-52s %4.1f/%.1f"
              % ("", self.etiqueta_total, obtenido, maximo))
        print("=" * 68)

        return 0 if abs(obtenido - maximo) < TOLERANCIA else 1


corrector = Corrector()


############################################################
# Problema 1a: extract_word_features
############################################################


def prueba_1a_0():
    import submission
    corrector.es_igual({"a": 2, "b": 1},
                       submission.extract_word_features("a b a"))
    corrector.es_igual({"me": 2, "gusta": 2, "y": 1},
                       submission.extract_word_features("me gusta y me gusta"),
                       "reseña en español")
    corrector.es_igual({"la": 1, "peor": 1, "película": 1, "del": 1, "año": 1},
                       submission.extract_word_features("la peor película del año"),
                       "acentos y ñ")


corrector.automatica("1a-0", prueba_1a_0, 5,
                     "conteo de palabras en casos básicos", segundos=2)


############################################################
# Problema 1b: learn_predictor
############################################################


def prueba_1b_0():
    import submission
    train = (("hello world", 1), ("goodnight moon", -1))
    validacion = (("hello", 1), ("moon", -1))
    weights = submission.learn_predictor(train, validacion,
                                         submission.extract_word_features,
                                         num_epochs=20, eta=0.01, verbose=False)
    corrector.es_mayor_que(0, weights.get("hello", 0.0), "peso de «hello»")
    corrector.es_menor_que(0, weights.get("moon", 0.0), "peso de «moon»")


corrector.automatica("1b-0", prueba_1b_0, 7,
                     "aprende el signo correcto en un caso mínimo", segundos=3)


def prueba_1b_2():
    import submission
    train = read_examples("resenas.train")
    validacion = read_examples("resenas.dev")
    extractor = submission.extract_word_features
    weights = submission.learn_predictor(train, validacion, extractor,
                                         num_epochs=20, eta=0.01, verbose=False)
    error_train = evaluate_predictor(
        train, lambda x: util.predecir(x, extractor, weights))
    error_val = evaluate_predictor(
        validacion, lambda x: util.predecir(x, extractor, weights))
    print("    error de entrenamiento = %.4f, error de validación = %.4f"
          % (error_train, error_val))
    corrector.es_menor_que(0.04, error_train, "error de entrenamiento")
    corrector.es_menor_que(0.30, error_val, "error de validación")


corrector.automatica("1b-2", prueba_1b_2, 10,
                     "error < 4% en entrenamiento y < 30% en validación",
                     segundos=40)


############################################################
# Problema 1c: extract_character_features
############################################################


def prueba_1c_0():
    import submission
    fe3 = submission.extract_character_features(3)
    corrector.es_igual({"hel": 1, "ell": 1, "llo": 1, "low": 1, "owo": 1,
                        "wor": 1, "orl": 1, "rld": 1},
                       fe3("hello world"))
    corrector.es_igual({"bue": 1, "uen": 1, "ena": 1, "nal": 1, "ala": 1,
                        "lap": 1, "ape": 1, "pel": 1, "eli": 1},
                       fe3("buena la peli"), "reseña en español")
    fe4 = submission.extract_character_features(4)
    corrector.es_igual({"lavi": 1}, fe4("la vi"))


corrector.automatica("1c-0", prueba_1c_0, 6,
                     "n-gramas de caracteres en casos básicos", segundos=2)


############################################################
# Problema 2a: error_analysis
############################################################

_PESOS_2A = {"buena": 1.0, "mala": -1.0, "peli": 0.2}
_EJEMPLOS_2A = [
    ("buena peli", 1),          # puntaje +1.2 -> correcto
    ("mala peli", -1),          # puntaje -0.8 -> correcto
    ("mala mala peli", 1),      # puntaje -1.8 -> error, margen -1.8
    ("buena buena", -1),        # puntaje +2.0 -> error, margen -2.0
    ("peli", -1),               # puntaje +0.2 -> error, margen -0.2
    ("palabra desconocida", -1),  # puntaje 0.0 -> predice +1, margen 0.0
]


def prueba_2a_0():
    import submission
    obtenido = submission.error_analysis(_EJEMPLOS_2A,
                                         submission.extract_word_features,
                                         _PESOS_2A, k=3)
    esperado = [("buena buena", -1, 1, -2.0),
                ("mala mala peli", 1, -1, -1.8),
                ("peli", -1, 1, -0.2)]
    corrector.es_igual(esperado, obtenido, "las 3 peores")
    todos = submission.error_analysis(_EJEMPLOS_2A,
                                      submission.extract_word_features,
                                      _PESOS_2A, k=10)
    corrector.es_igual(4, len(todos), "cantidad de reseñas mal clasificadas")
    corrector.es_igual(("palabra desconocida", -1, 1, 0.0), todos[3],
                       "puntaje 0 se predice como +1")


corrector.automatica("2a-0", prueba_2a_0, 6,
                     "las k peores reseñas, en orden y con su margen", segundos=3)


############################################################
# Problema 2b: caracteristicas_culpables
############################################################

_PESOS_2B = {"a": 1.0, "b": 1.0, "c": -1.0, "d": 0.5}


def prueba_2b_0():
    import submission
    extractor = submission.extract_word_features
    corrector.es_igual([("mala", -2.0)],
                       submission.caracteristicas_culpables(
                           "mala mala peli", 1, extractor, _PESOS_2A, k=3),
                       "una sola característica culpable")
    corrector.es_igual([("buena", -2.0)],
                       submission.caracteristicas_culpables(
                           "buena buena", -1, extractor, _PESOS_2A, k=3))
    corrector.es_igual([("a", -1.0), ("b", -1.0)],
                       submission.caracteristicas_culpables(
                           "a b c", -1, extractor, _PESOS_2B, k=5),
                       "empate resuelto alfabéticamente")
    corrector.es_igual([("a", -1.0)],
                       submission.caracteristicas_culpables(
                           "a b c", -1, extractor, _PESOS_2B, k=1),
                       "la lista se trunca a k")


corrector.automatica("2b-0", prueba_2b_0, 5,
                     "características que empujaron al lado equivocado",
                     segundos=3)


############################################################
# Problema 2c: mejor_n
############################################################


def prueba_2c_0():
    import submission
    train = read_examples("resenas.train")
    validacion = read_examples("resenas.dev")
    mejor, errores = submission.mejor_n(train, validacion, [1, 3, 6],
                                        num_epochs=20, eta=0.01)
    corrector.es_igual({1, 3, 6}, set(errores),
                       "el diccionario debe traer un error por cada n pedido")
    print("    errores de validación: " +
          ", ".join("n=%d -> %.4f" % (n, errores[n]) for n in sorted(errores)))
    corrector.es_mayor_que(0.35, errores[1],
                           "con n = 1 el modelo casi no distingue nada")
    corrector.es_menor_que(errores[1], errores[3], "error con n = 3")
    corrector.es_menor_que(0.32, errores[6], "error con n = 6")
    corrector.es_igual(6, mejor, "el n de menor error de validación")


corrector.automatica("2c-0", prueba_2c_0, 4,
                     "barrido de n y elección del mejor largo de n-grama",
                     segundos=120)


############################################################
# Problema 3a: margen suave
############################################################

# Conjunto fijo, con coordenadas exactas en binario: los márgenes no dependen
# del orden en que se sumen los términos.
_X_3A = [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [0.5, 0.0],
         [3.0, 0.0], [0.0, 3.0]]
_Y_3A = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0]


def prueba_3a_0():
    import submission
    X = np.array(_X_3A)
    y = np.array(_Y_3A)
    w = np.array([1.0, -1.0])

    # Con b = 0 los márgenes son 1, 1, 0, 0, -0,5, 3 y 3.
    corrector.es_igual([0, 1, 2, 3, 4],
                       list(submission.indices_vectores_de_soporte(X, y, w, 0.0)),
                       "vectores de soporte con b = 0")
    corrector.es_cercano(2.0 / 7.0,
                         submission.error_de_clasificacion(X, y, w, 0.0),
                         contexto="error con b = 0 (puntaje 0 predice +1)")

    # Con b = 0,5 los márgenes son 1,5, 0,5, 0,5, -0,5, -1, 3,5 y 2,5.
    corrector.es_igual([1, 2, 3, 4],
                       list(submission.indices_vectores_de_soporte(X, y, w, 0.5)),
                       "vectores de soporte con b = 0,5")
    corrector.es_cercano(2.0 / 7.0,
                         submission.error_de_clasificacion(X, y, w, 0.5),
                         contexto="error con b = 0,5")

    # La tolerancia se aplica sobre el margen, no sobre el puntaje.
    corrector.es_igual([0, 1, 2, 3, 4, 5, 6],
                       list(submission.indices_vectores_de_soporte(X, y, w, 0.0,
                                                                   tol=2.0)),
                       "una tolerancia grande deja a todos adentro")


corrector.automatica("3a-0", prueba_3a_0, 4,
                     "vectores de soporte y error de un (w, b) dado", segundos=5)


# Conjunto pequeño y separable: la solución de referencia deja J = 0,2639 con
# C = 1 y 500 iteraciones, así que la cota de abajo tiene holgura de sobra.
_X_3A1 = [[2.0, 0.0], [3.0, 1.0], [4.0, 1.0], [0.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
_Y_3A1 = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]


def prueba_3a1():
    import submission
    X = np.array(_X_3A1)
    y = np.array(_Y_3A1)
    C = 1.0
    w, b = submission.entrenar_svm_margen_suave(X, y, C, 500,
                                                util.paso_seguro(X, C))
    w = np.asarray(w, dtype=float)
    corrector.es_igual((2,), tuple(w.shape), "forma del vector de pesos")
    corrector.es_menor_que(0.30, util.objetivo_svm(X, y, w, b, C),
                           "J(w, b) tras 500 iteraciones con C = 1")
    corrector.es_igual(0.0, submission.error_de_clasificacion(X, y, w, b),
                       "error en un conjunto separable")
    corrector.es_mayor_que(0.0, float(w[0]), "peso de la característica positiva")
    corrector.es_menor_que(0.0, float(w[1]), "peso de la característica negativa")

    # Con cero iteraciones el punto de partida tiene que quedar intacto.
    w0, b0 = submission.entrenar_svm_margen_suave(X, y, C, 0, 0.1)
    corrector.es_cercano(0.0, float(np.abs(np.asarray(w0)).sum()),
                         contexto="con epocas = 0 el vector de pesos parte en 0")
    corrector.es_cercano(0.0, float(b0), contexto="con epocas = 0 el sesgo parte en 0")


corrector.automatica("3a-1", prueba_3a1, 4,
                     "entrenamiento de margen suave en un caso separable",
                     segundos=10)


############################################################
# Problema 3b: kernels
############################################################


def prueba_3b_0():
    import submission
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.0, -1.0, 2.0])          # a . b = 4
    corrector.es_cercano(25.0, submission.kernel_polinomial(a, b),
                         contexto="grado 2 y coef0 = 1 por omisión")
    corrector.es_cercano(125.0, submission.kernel_polinomial(a, b, grado=3),
                         contexto="grado 3")
    corrector.es_cercano(4.0,
                         submission.kernel_polinomial(a, b, grado=1, coef0=0.0),
                         contexto="grado 1 sin término constante es el producto punto")
    corrector.es_cercano(1.0,
                         submission.kernel_polinomial(np.array([2.0, 0.0]),
                                                      np.array([0.0, 3.0])),
                         contexto="vectores ortogonales")

    corrector.es_cercano(1.0,
                         submission.kernel_rbf(np.array([5.0, 5.0]),
                                               np.array([5.0, 5.0]), gamma=0.7),
                         contexto="un vector consigo mismo")
    corrector.es_cercano(3.726653172078671e-06,
                         submission.kernel_rbf(np.array([1.0, 2.0]),
                                               np.array([4.0, 6.0])),
                         contexto="gamma = 0,5 por omisión")
    corrector.es_cercano(0.1353352832366127,
                         submission.kernel_rbf(np.array([1.0, 0.0]),
                                               np.array([0.0, 1.0]), gamma=1.0))
    corrector.es_cercano(0.6065306597126334,
                         submission.kernel_rbf(np.array([3.0, 4.0]),
                                               np.array([0.0, 0.0]), gamma=0.02))


corrector.automatica("3b-0", prueba_3b_0, 5,
                     "kernel polinomial y kernel gaussiano", segundos=5)


def prueba_3b_1():
    import submission
    for d, dimension in ((1, 3), (2, 6), (3, 10), (5, 21)):
        psi = np.asarray(submission.mapa_polinomial_grado2(np.arange(1.0, d + 1.0)),
                         dtype=float)
        corrector.es_igual((dimension,), tuple(psi.shape),
                           "dimensión del mapa para d = %d" % d)

    for a, b in (([1.0, 2.0], [-3.0, 0.5]),
                 ([0.0, 0.0], [1.0, 1.0]),
                 ([2.0, -1.0, 0.5], [1.0, 1.0, -2.0])):
        va, vb = np.array(a), np.array(b)
        producto = float(np.dot(submission.mapa_polinomial_grado2(va),
                                submission.mapa_polinomial_grado2(vb)))
        corrector.es_cercano(submission.kernel_polinomial(va, vb, 2, 1.0),
                             producto,
                             contexto="psi(a) . psi(b) tiene que dar el kernel")


corrector.automatica("3b-1", prueba_3b_1, 2,
                     "el mapa explícito reproduce el kernel de grado 2",
                     segundos=5)


def _vectores_reproducibles(semilla: int, cantidad: int, d: int) -> List[np.ndarray]:
    """Vectores deterministas, iguales en cualquier versión de Python."""
    estado = semilla * 2654435761 % (2 ** 32)
    vectores = []
    for _ in range(cantidad):
        fila = []
        for _ in range(d):
            estado = (1103515245 * estado + 12345) % (2 ** 31)
            fila.append((estado % 2001 - 1000) / 250.0)
        vectores.append(np.array(fila))
    return vectores


def prueba_3b_2():
    import submission
    vectores = _vectores_reproducibles(23, 12, 3)

    for a in vectores:
        corrector.es_cercano(1.0, submission.kernel_rbf(a, a, gamma=0.3),
                             contexto="el kernel gaussiano vale 1 en la diagonal")
    for a, b in zip(vectores, vectores[1:]):
        corrector.es_cercano(submission.kernel_polinomial(a, b, 4, 0.5),
                             submission.kernel_polinomial(b, a, 4, 0.5),
                             contexto="un kernel es simétrico")
        corrector.es_cercano(submission.kernel_rbf(a, b, gamma=0.9),
                             submission.kernel_rbf(b, a, gamma=0.9),
                             contexto="un kernel es simétrico")
        corrector.es_cercano(float(np.dot(a, b)),
                             submission.kernel_polinomial(a, b, grado=1, coef0=0.0),
                             contexto="grado 1 sin constante es el producto punto")
        corrector.es_cercano(
            submission.kernel_polinomial(a, b, 2, 1.0),
            float(np.dot(submission.mapa_polinomial_grado2(a),
                         submission.mapa_polinomial_grado2(b))),
            contexto="el mapa explícito no reproduce el kernel")

    # Un kernel legítimo produce matrices de Gram semidefinidas positivas.
    for nombre, k in (("polinomial", lambda a, b: submission.kernel_polinomial(a, b, 3, 1.0)),
                      ("gaussiano", lambda a, b: submission.kernel_rbf(a, b, gamma=0.4))):
        G = np.array([[k(a, b) for b in vectores] for a in vectores])
        propios = np.linalg.eigvalsh((G + G.T) / 2.0)
        corrector.es_verdadero(
            float(propios.min()) >= -1e-8 * max(1.0, float(propios.max())),
            "la matriz de Gram del kernel %s no es semidefinida positiva" % nombre)

    # El kernel gaussiano decrece con la distancia.
    origen = np.zeros(3)
    anteriores = None
    for escala in (0.5, 1.0, 2.0, 4.0):
        valor = submission.kernel_rbf(origen, np.full(3, escala), gamma=0.6)
        corrector.es_verdadero(0.0 < valor <= 1.0,
                               "el kernel gaussiano tiene que quedar en (0, 1]")
        if anteriores is not None:
            corrector.es_menor_que(anteriores, valor,
                                   "el kernel gaussiano no decrece con la distancia")
        anteriores = valor


corrector.automatica("3b-2", prueba_3b_2, 2,
                     "propiedades que todo kernel válido tiene que cumplir",
                     segundos=10)


############################################################

# La pone en alto el corrector del curso al importar este archivo.
ES_CORRECCION_DEL_CURSO = False

if __name__ == "__main__":
    seleccion = sys.argv[1] if len(sys.argv) > 1 else None
    codigo = corrector.correr(seleccion)
    if seleccion is None and not ES_CORRECCION_DEL_CURSO:
        print()
        print("Este subtotal cubre 60 de los 100 puntos de la tarea. Los 40")
        print("restantes los pone el corrector del curso, que corre estas")
        print("mismas pruebas, agrega pruebas ocultas sobre las mismas")
        print("funciones y revisa la sección de respuestas de submission.py:")
        print("la reseña del Problema 2d y los números del Problema 3c. Ese")
        print("puntaje también se reparte solo.")
    sys.exit(codigo)
