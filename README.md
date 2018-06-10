# ojoDeDios

# Introducción:

Este es un proyecto realizado para el curso tópicos especiales en telemática que trata de poner en practica los conocimiento adquiridos en el mini-curso de aws por PLS. La problemática que se busca atacar es la de encontrar personas sospechosas o personas objetivo dentro de una grabación de video. Todo esto a partir de la imagen de una cara de dicha persona.

  Integrantes:
  
    - Juan Camilo Henao Salazar.
    
    - Juan Fernando Rincón Cardeño.
    
    - Juan Diego Zuluaga Gallo.



# Requerimientos:

Este proyecto fue realizado utilizando python3.5, opencv3.4 y rekognition de aws.

Las bibliotecas utilizadas para python fueron: Numpy,cv2 y boto3. Todas estas se pueden instalar directamente con el instalador de paquetes de python(pip). 

Además se requiere de un cliente de amazon para correr el proyecto. Para esto es necesario utilizar el comando "aws configure" y dar todas las credenciales de la cuenta.

```
sudo pip3.5 install opencv-python
sudo pip3.5 install numpy
sudo pip3.5 install boto3
aws configure
```

# Funcionamiento:

Inicialmente, antes de realizar peticiones a la API de amazon, se tiene una primera parte de pre-procesamiento. 

Esta consiste en realizar una detección y reconocimiento facial localmente, antes de envíar una petición al servidor de amazon.
Utilizando Opencv detectamos y reconocemos una cara. Si se determina que ésta, pertenece a la persona buscada, entonces aws rekognition se encarga de dar la última palabra.

Para entrenar el modelo de reconocimiento de Opencv es necesario lo siguiente:

Dentro de la carpeta "imagesForTraining" crear una carpeta con el nombre de la persona que va a entrenar, y dentro de ella poner todas las imágenes de la cara de dicha persona. Luego corremos el script de python recognizeFacesTrain.py, con python3.5

```
python3.5 recognizeFacesTrain.py
```
Esto genera un archivo llamado face-trainner.yml que se encuentra dentro de la carpeta recognizers.

Después de esto, se podrá corrar el programa principal llamado findSuspect.py: 

```
python3.5 findSuspect.py -t "path del imagen target" -v 0
```
Este programa recibe dos parámetros, el primetro es el path de la imagen de la persona a buscar, y el segundo es el path del video en el cuál se desea buscar dicha persona. Si en el parámetro -v ingresamos un 0 en vez de un path, este utilizará la webcam de la computadora
