# Tarea 3
## Isaac Rodríguez Bribiesca

* Uso:

    * Ejecutar archivo run.py
    * Ejecutar "python run.py --help" para imprimir información sobre los parámetros

* Parámetros:

    * Tipo de tamaño de paso: fijo, hess, back [-s]
    * Tipo de punto incial x a usar: const, rand [-p]
    * Valor parámetro lambda de función: 1, 100, 1000 [-l]
    * Metodo de optimizacion a usar: newton, gd [-m]

* Ejemplos:

Ejecución de Newton con lambda = 1000, tamaño de paso fijo con valor 1 y punto x inicial fijo [1, 1, ..., 1]

  * python run.py -s fijo -p const -m newton -l 1000

Ejecución de Gradiente Descendiente con lambda = 1, tamaño de paso con matriz Hessiana y punto x inicial fijo [1, 1, ..., 1]

  * python run.py -s hess -p const -m gd -l 1
