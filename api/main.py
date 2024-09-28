from flask import Flask, render_template, redirect, request
import numpy as np
from PIL import Image
from projects.athlete_progan.eval import gen_images

app = Flask(__name__)

@app.route("/")
def start():
    return redirect("/home")

@app.route("/home")
def home():
    return render_template("pages/home.html")

@app.route("/projects")
def projects():
    return render_template("pages/projects.html")

@app.route("/projects/athlete_progan")
def athlete_progan():
    return render_template("pages/athlete_progan.html")

@app.route("/projects/map_generator")
def map_generator():
    return render_template("pages/map_generator.html")

@app.route("/projects/football_simulator")
def football_simulator():
    return render_template("pages/football_simulator.html")

@app.route("/experience")
def experience():
    return render_template("pages/experience.html")

@app.route("/skills")
def skills():
    return render_template("pages/skills.html")

@app.route("/resume")
def resume():
    return render_template("pages/resume.html")

@app.route("/contact")
def contact():
    return render_template("pages/contact.html")
