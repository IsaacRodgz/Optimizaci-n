# Tarea 9
## Isaac Rodríguez Bribiesca

* Uso:

    * Ejecutar archivo run.py
    * Ejecutar "python run.py --help" para imprimir información sobre los parámetros

* Parámetros:

    * Función a minimizar: rosenbrock, wood [-f]
    * Método de optimización Cuasi-Newton: dfp, bfgs [-m]

* Ejemplos:

  * Ejecución de DFP, aplicado a función Rosenbrock

    * python run.py -m dfp -f rosenbrock

  * Ejecución de BFGS, aplicado a función Wood.

    * python run.py -m bfgs -f wood
