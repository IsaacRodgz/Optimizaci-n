# Tarea 5
## Isaac Rodríguez Bribiesca

### Rosenbrock

* Uso:

    * Para pruebas con función Rosenbrock, ejecutar archivo run.py
    * Ejecutar "python run.py --help" para imprimir información sobre los parámetros

* Parámetros:

    * Tipo de punto incial x a usar: const, rand [-p]
    * Metodo de optimizacion a usar: dogleg, lstr [-m]

* Ejemplos:

Ejecución de Dogleg y punto x inicial fijo [-1.2, 1, ..., -1.2, 1]

  * python run.py -p const -m dogleg

  Ejecución de LSTR y punto x inicial fijo [-1.2, 1, ..., -1.2, 1]

    * python run.py -p const -m lstr

### Segmentación de imágenes

* Uso:

  * Para pruebas con segmentación de imagen, ejecutar archivo segment.py
  * Ejecutar "python segment.py --help" para imprimir información sobre los parámetros

  * Parámetros:

      * Número de funciones gaussianas a usar [-s]
      * Metodo de optimizacion a usar: dogleg, lstr [-m]
      * Valor de parámetro sigma [-v]
      * Nombre de folder que contiene histogramas [-f]
      * Nombre de imagen a segmentar [-i]
      * Nombre de archivo de imagen segmentada a crear [-o]
      * Opción para segmentar imagen usando histogramas originales: no, yes [-t]

  * Ejemplos:

  Ejecución de Dogleg con 10 gaussianas, sigma = 5, y segmentacion de iamgen "grave.bmp"

    * python segment.py -o grave_segmented.png -i grave.bmp -f histograms -m dogleg -s 10 -v 5

  Ejecución de lstr con 10 gaussianas, sigma = 10, y segmentacion de iamgen "grave.bmp"

    * python segment.py -o grave_segmented.png -i grave.bmp -f histograms -m lstr -s 10 -v 10
