# Tarea 6
## Isaac Rodríguez Bribiesca

* Uso:

    * Ejecutar archivo run.py
    * Ejecutar "python run.py --help" para imprimir información sobre los parámetros

* Parámetros:

    * Dimensión de la matriz del problema [-d]
    * Tipo de punto incial x a usar: const, rand [-p]
    * Valor de parámetro lambda [-l]

* Ejemplos:

Ejecución de gradiente conjugado, para matriz de dimensión 128*128, valor de regularización lambda = 100 y punto x inicial fijo [1, 1, ..., 1].

  * python run.py -d 128 -l 100
