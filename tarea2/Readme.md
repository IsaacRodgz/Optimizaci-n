# Tarea 2
## Isaac Rodríguez Bribiesca

* Uso:

    * Ejecutar archivo test.py

* Parámetros:

    * Tipo de tamaño de paso: fijo, hess, back [-s]
    * Tipo de punto incial x a usar: const, rand [-p]
    * Función de prueba a usar: ros2, ros100, wood [-f]

* Ejemplo:

Ejecución de gradiente descendiente con función Rosenberg n = 2, tamaño de paso fijo y punto x inicial [-1.2, 1] fijo

  * python test.py -f ros2 -s fijo -p const

Ejecución de gradiente descendiente con función Wood, tamaño de paso calculado con Backtracking y punto x inicial aleatorio

  * python test.py -f wood -s back -p rand
