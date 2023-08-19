"""
The flask application package.
"""
# Importar la clase Flask del módulo flask
from flask import Flask
# Crear una instancia de la clase Flask y asignarla a la variable 'app'
app = Flask(__name__)
# Importar las vistas (rutas y controladores) del archivo FlaskProyecto.views
import FlaskProyecto.views
