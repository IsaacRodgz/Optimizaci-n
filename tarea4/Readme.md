# Tarea 4
## Isaac Rodríguez Bribiesca

* Uso:

    * Ejecutar archivo run.py
    * Ejecutar "python run.py --help" para imprimir información sobre los parámetros

* Parámetros:

    * Tipo de tamaño de paso: cubic, barzilai, zhang [-s]
    * Tipo de punto incial x a usar: const, rand [-p]
    * Función a optimizar: rosenbrock, wood, mnist [-p]

* Ejemplos:

Ejecución de gradiente descendiente para optimizar regresión logística con el dataset de MNIST, con tamaño de paso calculado mediante Barzilai-Borwein y punto x inicial fijo [1, 1, ..., 1].

  * python run.py -p const -s barzilai -f mnist

Ejecución de gradiente descendiente para optimizar la función Wood,  con tamaño de paso calculado mediante Zhang-Hager y punto x inicial fijo [-3, -1, -3, -1]

  * python run.py -p const -s zhang -f wood
