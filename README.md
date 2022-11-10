# Proyecto PDI

## Detección Lengua de Señas

## Descripción
Proyecto para el ramo de Procesamiento Digital de Imágenes (ELO-328), semestre 2022-2, Campus San Joaquín. El proyecto esta enfocado en utilizar redes neuronales para clasificar imágenes extraídas de video en directo con el fin de predecir frases de lenguas de señas chilenas.

## Instalación
En primer lugar se debe clonar el proyecto mediante el siguiente comando:
```
git clone https://gitlab.com/grupo-g-elo329/proyecto-pdi.git
cd proyecto-pdi
```

O en caso contrario descargar los archivos via zip.\
A continuación se debe crear un ambiente virtual de python para trabajar en un entorno cerrado con las dependencias del algoritmo de detección.

```
pip install virtualenv
virtualenv my_env
```
Posteriormente se debe activar el entorno virtual e instalar los paquetes de Python necesarios para correr el proyecto.
```
source my_venv/bin/activate
pip install -r requirements.txt
```
También es valido crear un ambiente de [Conda](https://docs.conda.io/en/latest/) en Python e instalar en este las dependencias.


## Uso
Para levantar el servidor de Flask se debe ejecutar el archivo `videostreaming_web.py` con
```
python videostreaming_web.py
```
Es probable que el sistema operativo solicite los permisos para utilizar la cámara del equipo.
Para visualizar la pagina acceda a https://127.0.0.1:5000 desde la maquina local o https://\<ip_del_servidor>:5000 para acceder de forma remota desde la red local.
