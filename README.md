[![codecov](https://codecov.io/gh/frapercan/python-poetry-template/graph/badge.svg?token=hqzrADVeRy)](https://codecov.io/gh/frapercan/python-poetry-template)
[![PyPI version](https://badge.fury.io/py/python-poetry-template.svg)](https://badge.fury.io/py/python-poetry-template)
[![Documentation Status](https://readthedocs.org/projects/python-poetry-template/badge/?version=latest)](https://python-poetry-template.readthedocs.io/en/latest/?badge=latest)
![Linting Status](https://github.com/frapercan/python-poetry-template/actions/workflows/test-lint.yml/badge.svg?branch=main)

# Python Poetry Template

Este proyecto, "Python Poetry Template", es una plantilla versátil para proyectos de Python, diseñada para facilitar la gestión de dependencias, la publicación de módulos, el análisis estático de código, las pruebas y la generación de documentación. Utilizando herramientas modernas como Poetry, PyPI, Flake8, Pytest, Coverage y Sphinx con tema Read The Docs, esta plantilla está orientada a maximizar la eficiencia y calidad en el desarrollo de software en Python.

## Herramientas Utilizadas

- **Poetry:** Gestión de dependencias y empaquetado en Python.
- **PyPI:** Plataforma para la publicación de paquetes Python.
- **Flake8:** Análisis estático de código para cumplimiento de estilo.
- **Pytest y Coverage:** Framework de pruebas y medición de cobertura de código.
- **Sphinx (Tema RTD):** Generación de documentación del proyecto.

## Configuración del Proyecto

Para configurar el entorno de desarrollo:

1. Clone el repositorio: `git clone [URL del repositorio]`.
2. Instale Poetry si aún no está instalado en su sistema.
3. Ejecute `poetry install` para instalar las dependencias del proyecto.

## Renombrar el contenedor principal de código
El código se encuentra por defecto bajo el directorio **`python_poetry_template`**, debemos renombrarla, al nombre deseado.

## Configuración de *pyproject.toml* 
Asegúrate de que tu archivo **`pyproject.toml`** esté correctamente configurado. Esto incluye especificar el nombre del paquete, la versión, la descripción, los autores y cualquier otra información relevante que PyPI necesite para mostrar tu paquete correctamente.

Por ejemplo:
```toml
[tool.poetry]
name = "nombre-de-tu-paquete"
version = "0.1.0"
description = "Una breve descripción de tu paquete"
authors = ["Tu Nombre <tu_email@example.com>"]

...

[tool.taskipy.tasks]
html_docs = "make html -C docs"
lint = "poetry run flake8 <ruta-directorio-principal>"
coverage = "poetry run coverage run -m --source=<ruta-directorio-principal> pytest tests && poetry run coverage report -m"

```

## Configuración del la documentación

En el fichero de configuración de sphinx **`docs/source/conf.py`** debemos configurar con los valores de nuestro proyecto.

```python
project = 'python_poetry_template'
copyright = '2023, frapercan'
author = 'frapercan'
```

también debemos configurar `docs/source/index.rst` haciendo referencia a los ficheros .rst que gestionan nuestros módulos de programación , teniendo `docs/source/template.rst` cómo ejemplo de muestra.

En caso de dudas atender a la documentación oficial de sphinx.





## Comandos

### Generar Documentación

```bash
poetry run task html_docs
```
Este comando construye la documentación del proyecto con Sphinx y el tema RTD, ubicándola en **docs/build**.

### Análisis Estático de Código (Linting)
```bash
poetry run task lint
```
Ejecuta Flake8 para análisis estático del código, ayudando a identificar problemas de estilo y errores.

### Cobertura de Pruebas
```bash
poetry run task coverage
```
Realiza pruebas con Pytest y genera un informe de cobertura con Coverage.

## Publicación del Paquete
Para compilar y publicar tu paquete Python utilizando Poetry, sigue estos pasos:

### Compilación del Paquete
Antes de publicar, es una buena práctica compilar el paquete para asegurarse de que todo esté configurado correctamente. Para compilar tu paquete, ejecuta:
```bash
poetry build
```
Este comando generará los archivos de distribución en los formatos de archivo Wheel y Source. Los archivos se crearán en el directorio **`dist/`** de tu proyecto.

### Publicación en PyPI
Una vez que hayas compilado tu paquete, puedes publicarlo en [PyPI](https://pypi.org) para que otros puedan instalarlo fácilmente. Asegúrate de tener una cuenta en PyPI y haber configurado tus credenciales de autenticación.
```bash
poetry publish
```
Este comando te pedirá tus credenciales de PyPI (si no están ya almacenadas) y luego subirá los archivos de distribución a PyPI.

# Python Poetry Template

Este proyecto, "Python Poetry Template", es una plantilla versátil para proyectos de Python, diseñada para facilitar la gestión de dependencias, la publicación de módulos, el análisis estático de código, las pruebas y la generación de documentación. Utilizando herramientas modernas como Poetry, PyPI, Flake8, Pytest, Coverage y Sphinx con tema Read The Docs, esta plantilla está orientada a maximizar la eficiencia y calidad en el desarrollo de software en Python.

## Herramientas Utilizadas

- **Poetry:** Gestión de dependencias y empaquetado en Python.
- **PyPI:** Plataforma para la publicación de paquetes Python.
- **Flake8:** Análisis estático de código para cumplimiento de estilo.
- **Pytest y Coverage:** Framework de pruebas y medición de cobertura de código.
- **Sphinx (Tema RTD):** Generación de documentación del proyecto.

## Configuración del Proyecto

Para configurar el entorno de desarrollo:

1. Clone el repositorio: `git clone [URL del repositorio]`.
2. Instale Poetry si aún no está instalado en su sistema.
3. Ejecute `poetry install` para instalar las dependencias del proyecto.

## Renombrar el contenedor principal de código
El código se encuentra por defecto bajo el directorio **`template`**, debemos renombrarla, al nombre deseado.

## Configuración de *pyproject.toml* 
Asegúrate de que tu archivo **`pyproject.toml`** esté correctamente configurado. Esto incluye especificar el nombre del paquete, la versión, la descripción, los autores y cualquier otra información relevante que PyPI necesite para mostrar tu paquete correctamente.

Por ejemplo:
```toml
[tool.poetry]
name = "nombre-de-tu-paquete"
version = "0.1.0"
description = "Una breve descripción de tu paquete"
authors = ["Tu Nombre <tu_email@example.com>"]

...

[tool.taskipy.tasks]
html_docs = "make html -C docs"
lint = "poetry run flake8 <ruta-directorio-principal>"
coverage = "poetry run coverage run -m --source=<ruta-directorio-principal> pytest tests && poetry run coverage report -m"

```

## Configuración del la documentación

En el fichero de configuración de sphinx **`docs/source/conf.py`** debemos configurar con los valores de nuestro proyecto.

```python
project = 'template'
copyright = '2023, frapercan'
author = 'frapercan'
```

también debemos configurar `docs/source/index.rst` haciendo referencia a los ficheros .rst que gestionan nuestros módulos dep programación , teniendo `docs/source/template.rst` cómo ejemplo de muestra.

En caso de dudas atender a la documentación oficial de sphinx.





## Comandos

### Generar Documentación

```bash
poetry run task html_docs
```
Este comando construye la documentación del proyecto con Sphinx y el tema RTD, ubicándola en **docs/build**.

### Análisis Estático de Código (Linting)
```bash
poetry run task lint
```
Ejecuta Flake8 para análisis estático del código, ayudando a identificar problemas de estilo y errores.

### Cobertura de Pruebas
```bash
poetry run task coverage
```
Realiza pruebas con Pytest y genera un informe de cobertura con Coverage.

## Publicación del Paquete
Para compilar y publicar tu paquete Python utilizando Poetry, sigue estos pasos:

### Aumentar la versión del proyecto.
```bash
poetry version minor
```

```bash
poetry version major
```

### Compilación del Paquete
Antes de publicar, es una buena práctica compilar el paquete para asegurarse de que todo esté configurado correctamente. Para compilar tu paquete, ejecuta:
```bash
poetry build
```
Este comando generará los archivos de distribución en los formatos de archivo Wheel y Source. Los archivos se crearán en el directorio **`dist/`** de tu proyecto.

### Publicación en PyPI
Una vez que hayas compilado tu paquete, puedes publicarlo en [PyPI](https://pypi.org) para que otros puedan instalarlo fácilmente. Asegúrate de tener una cuenta en PyPI y haber configurado tus credenciales de autenticación.
```bash
poetry publish
```
Este comando te pedirá tus credenciales de PyPI (si no están ya almacenadas) y luego subirá los archivos de distribución a PyPI.

### Configurar credenciales de PiPy
Para configurar la github actions, debemos de añadir la clave secreta en la configuración del repositorio bajo el nombre `PYPI_API_TOKEN`.
Este token lo sacamos tras haber configurado nuestra cuenta de pipy.

## Configurar Codecov y ReadTheDocs
A través de la página de la aplicación [CodeCov](https://app.codecov.io/) y [ReadTheDocs](https://readthedocs.org/) podemos configurar los repositorios bajo nuestro usuario de github para que sean monitorizados por estos servicios.

## Configurar los badges:
Tomando las primeras lineas de este README.md de muestra, podemos reemplazar los valores de nuestro template, por los de nuestro proyecto deseado de forma intuitiva.

