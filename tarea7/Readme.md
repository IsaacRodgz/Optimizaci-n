# Tarea 7
## Isaac Rodríguez Bribiesca

* Uso:

    * Ejecutar archivo run.py
    * Ejecutar "python run.py --help" para imprimir información sobre los parámetros

* Parámetros:

    * Tipo de punto incial x a usar: const, rand [-p]
    * Función a minimizar: rosenbrock, wood [-f]
    * Método para calcular parámetro beta: fr, pr, hs [-b]

* Ejemplos:

  * Ejecución de gradiente conjugado no lineal con método Fletcher-Reeves, aplicado a función Rosenbrock y punto x inicial fijo [1, 1, ..., 1].

    * python run.py -b fr -f rosenbrock -p const

  * Ejecución de gradiente conjugado no lineal con método Fletcher-Reeves Polak-Ribiere, aplicado a función Wood y punto x inicial aleatorio.

    * python run.py -b pr -f wood -p rand
