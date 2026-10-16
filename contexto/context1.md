# Contexto 1: Solución Tarea Modelos No Lineales y SVM
**Fecha de registro:** 2026-10-16

## 1. Instrucciones
* Desarrollar la solución de cada uno de los pasos solicitados en `enunciado.pdf`.
* Completar el único archivo entregable: `submission.py`.
* Implementar las doce funciones del código respetando las restricciones de bibliotecas (Problemas 1 y 2: biblioteca estándar de Python; Problema 3: numpy exclusivamente, sin scikit-learn).
* Problema 1:
  - 1a: `extract_word_features(x)`: extracción de conteo de palabras (bolsa de palabras).
  - 1b: `learn_predictor(...)`: entrenamiento SGD con pérdida hinge.
  - 1c: `extract_character_features(n)`: n-gramas de caracteres sin espacios ni tabuladores.
* Problema 2:
  - 2a: `error_analysis(...)`: reseñas peor clasificadas (solo mal clasificadas), ordenadas de margen menor a mayor.
  - 2b: `caracteristicas_culpables(...)`: características con aporte estrictamente negativo ordenadas por aporte ascendente y luego alfabéticamente.
  - 2c: `mejor_n(...)`: barrido de largos de n-grama devolviendo (mejor, errores).
  - 2d: `P2_RESENA`: reseña negativa de al menos 4 palabras fuera del vocabulario de entrenamiento donde 6-gramas prediga negativo.
* Problema 3:
  - 3a: `entrenar_svm_margen_suave`, `indices_vectores_de_soporte`, `error_de_clasificacion`.
  - 3b: `kernel_polinomial`, `kernel_rbf`, `mapa_polinomial_grado2`.
  - 3c: Constantes de respuesta (`SVM_VECTORES_DE_SOPORTE`, `SVM_ERROR_ENTRENAMIENTO`, `SVM_ERROR_VALIDACION`, `SVM_GRADO2_ERROR_VALIDACION`, `SGD_ERROR_VALIDACION`).
* Validar que todas las pruebas pasen con `python grader.py` (60/60) y que los scripts principales se ejecuten sin errores.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar y validar todas las funciones y respuestas de la Tarea de Modelos No Lineales y SVM en `submission.py`.
* **Fases/Hitos principales:**
  1. Fase 1: Implementar Problema 1 (1a, 1b, 1c) y verificar con grader.py.
  2. Fase 2: Implementar Problema 2 (2a, 2b, 2c) y verificar con grader.py.
  3. Fase 3: Implementar Problema 3 (3a, 3b) y verificar con grader.py.
  4. Fase 4: Ejecutar scripts (`python submission.py`, `python submission.py ngramas`, `python submission.py svm`), formular P2_RESENA (2d) y completar constantes (3c).
  5. Fase 5: Verificación final integral con `grader.py`.
* **Entregables:**
  - `submission.py` completado.

## 3. Referencias
* **Path del plan original:** `enunciado.pdf`
