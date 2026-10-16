# Tarea: Modelos No Lineales y SVM

Ciencia de Datos — Universidad Católica del Norte

Acompaña la sesión de modelos no lineales: predictor lineal, características
no lineales, margen máximo y kernels. La parte de programación de los Problemas
1 y 2 conviene cerrarla de una sentada; el Problema 3 puede quedar planteado y
terminarse después.

## Qué hay en esta carpeta

| Archivo | Para qué sirve |
|---|---|
| `enunciado.pdf` | El enunciado completo de la tarea. Léelo primero. |
| `submission.py` | **El único archivo que entregas**: la sección de respuestas del comienzo (la reseña del Problema 2d y los números del Problema 3c) y las doce funciones que debes programar. |
| `grader.py` | Corrector automático. Corre sin conexión a internet. |
| `util.py` | Vectores dispersos, lectura del corpus, la representación numérica del Problema 3 y utilidades de análisis. Ya está completo: no lo modifiques. |
| `requirements.txt` | La única dependencia externa: `numpy`. |
| `datos/resenas.train` | 3.554 reseñas de entrenamiento, una por línea, con formato `<etiqueta> <texto>`. |
| `datos/resenas.dev` | 3.554 reseñas de validación, mismo formato. Solo para medir, nunca para entrenar. |

Los archivos de `datos/` se resuelven relativo a `util.py`, no al directorio de
trabajo: por eso el corrector funciona aunque lo llames desde otra carpeta.

## Requisitos

Python 3.9 o superior y `numpy`:

```
pip install -r requirements.txt
```

Los Problemas 1 y 2 usan **solo la biblioteca estándar**: ahí no debes importar
`numpy` ni nada más. El Problema 3 se resuelve con `numpy` y con nada más:
`scikit-learn` está prohibido en toda la tarea, incluso para comparar.

Funciona igual en Linux, macOS y Windows. En Windows el corrector desactiva el
límite de tiempo por prueba (el sistema no ofrece la señal que usa), pero corre
y puntúa exactamente igual.

## Antes de empezar

`COMO-TRABAJAR.pdf` trae las instrucciones generales: qué instalar, cómo abrir
una terminal en esta carpeta, qué significa el puntaje del corrector y los
errores más frecuentes. Sirve para todas las tareas del curso.

Un detalle que conviene saber de entrada: en Windows el intérprete se llama
`python` (o `py`), y en Linux y macOS suele ser `python3`. Los comandos de acá
usan `python`: si tu sistema responde «command not found», prueba con
`python3`.

## Cómo trabajar

1. Abre `submission.py` y completa las doce funciones de la sección de CÓDIGO.
   Tu código va entre los comentarios `BEGIN_TU_CODIGO` y `END_TU_CODIGO`.

   - Problema 1: `extract_word_features`, `learn_predictor`,
     `extract_character_features`.
   - Problema 2: `error_analysis`, `caracteristicas_culpables`, `mejor_n`.
   - Problema 3: `entrenar_svm_margen_suave`, `indices_vectores_de_soporte`,
     `error_de_clasificacion`, `kernel_polinomial`, `kernel_rbf`,
     `mapa_polinomial_grado2`.

2. Ejecuta el corrector desde esta carpeta:

   ```
   python grader.py
   ```

   Para probar una sola parte:

   ```
   python grader.py 1b-2
   ```

   Los códigos válidos son `1a-0`, `1b-0`, `1b-2`, `1c-0`, `2a-0`, `2b-0`,
   `2c-0`, `3a-0`, `3a-1`, `3b-0`, `3b-1` y `3b-2`. Si escribes cualquier otra
   cosa, el corrector te lo dice en vez de informar un falso éxito con 0 de 0
   puntos.

3. Cuando las funciones estén listas, corre el programa completo sobre el
   corpus real:

   ```
   python submission.py            # entrena con palabras, imprime las cinco
                                    # peores reseñas con sus culpables y deja
                                    # pesos.txt y analisis-error.txt
   python submission.py ngramas    # barrido de n = 2..7
   python submission.py svm        # barrido de C, kernel de grado 2 y
                                    # comparación con el SGD sin regularizar
   ```

4. Vuelve al comienzo de `submission.py` y completa la sección de RESPUESTAS:
   la reseña del Problema 2d y las cinco constantes del Problema 3c, que salen
   de lo que imprime `python submission.py svm`.

## Sobre el puntaje que informa el corrector

`grader.py` reparte **60 de los 100 puntos** de la tarea: las pruebas visibles
sobre el código de `submission.py`. Hay al menos una por cada función que
programas y, cuando fallan, el mensaje dice exactamente qué se esperaba y qué
se obtuvo, así que sirven para depurar.

Por eso la última fila del informe dice `SUBTOTAL` y no `TOTAL`. Los **40
puntos restantes** los pone el corrector del curso, que corre estas mismas
pruebas y agrega dos cosas más:

- **pruebas ocultas** sobre las mismas funciones, con casos de escala, de borde
  y de consistencia que no vienen en tu copia. Están para verificar que tu
  implementación no quedó calzada a los ejemplos del enunciado.
- la **revisión de la sección de respuestas**: la reseña del Problema 2d y los
  números del Problema 3c.

Esos 40 puntos también se reparten solos. En toda la tarea no hay ninguna parte
que se corrija a mano.

Llegar a 60/60 acá es condición necesaria para el puntaje completo, no
suficiente: conviene probar tus funciones con casos propios además de los del
corrector.

## Sobre los datos

`datos/resenas.train` y `datos/resenas.dev` son un conjunto de referencia
público de reseñas de cine en inglés, con etiquetas de calidad, distribuido con
fines docentes. Van incluidos en esta carpeta: el corrector no necesita
conexión a internet en ningún momento.

Los dos léxicos de sentimiento que usa el Problema 3 no vienen de un
diccionario externo: `util.py` los deriva del conjunto de entrenamiento con una
regla determinista que está documentada en el mismo archivo, así que la matriz
de características es idéntica para todo el curso.

## Qué entregar

Un solo archivo: **`submission.py`**, con las doce funciones implementadas y
todas las constantes de la sección de respuestas completadas.

No entregues nada más: ni la carpeta `datos/`, ni `util.py`, ni los archivos
que genera el programa (`pesos.txt`, `analisis-error.txt`).
