# Tarea 3
## Isaac Rodríguez Bribiesca

* Uso:

    * Ejecutar archivo run.py

* Parámetros:

    * Tipo de tamaño de paso: fijo, hess, back [-s]
    * Tipo de punto incial x a usar: const, rand [-p]
    * Valor parámetro lambda de función: 1, 100, 1000 [-l]

* Ejemplo:

Ejecución de Newton con lambda = 100, tamaño de paso fijo y punto x inicial [1, 1, ..., 1]

  * run.py -l 100 -s fijo -p const

Ejecución de Newton con lambda = 1, tamaño de paso con matriz hessiana y punto x inicial aleatorio

  * run.py -l 1 -s hess -p rand
