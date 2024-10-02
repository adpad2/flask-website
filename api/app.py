from flask import Flask, render_template, redirect, request, jsonify

import numpy as np
from PIL import Image
from api.projects.athlete_progan.eval import gen_images, TEAMS, TEAM_NAMES, BUILDS, SKIN_TONES
from api.init import init_generator
import io
import base64
import threading
import time
import requests

app = Flask(__name__)

netG = init_generator('api/projects/athlete_progan/generator.pth')

"""
def ping():
    # Continuously ping the server to prevent the site from spinning down
    url = "https://flask-website-8fy3.onrender.com"
    while True:
        try:
            response = requests.get(url)
        except Exception as e:
            print(f"Failed to ping {url}: {e}")
        time.sleep(600)  # Ping every 10 minutes

# Start the ping function in a background thread
threading.Thread(target=ping, daemon=True).start()
"""

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
    zipped_teams = zip(['any'] + TEAMS, ['Any'] + TEAM_NAMES)
    zipped_builds = zip(['any'] + BUILDS, ['Any'] + [build.replace('-', ' ').title() for build in BUILDS])
    zipped_skin_tones = zip(['any'] + SKIN_TONES, ['Any'] + [skin_tone.replace('-', ' ').title() for skin_tone in SKIN_TONES])
    return render_template("pages/athlete_progan.html", zipped_teams=zipped_teams,
                           zipped_builds=zipped_builds, zipped_skin_tones=zipped_skin_tones)

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

@app.route('/generate', methods=['POST'])
def generate():
    team = request.form.get('team')
    skin_tone = request.form.get('skin-tone')
    build = request.form.get('build')
    selected_image_ids = request.form.getlist('ids[]')

    images = gen_images(netG, team, skin_tone, build, len(selected_image_ids))
    images = images.detach().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    images = (images * 0.5 + 0.5) * 255
    images = images.astype(np.uint8)

    images_dict = {}
    for i, selected_image_id in enumerate(selected_image_ids):
        pil_image = Image.fromarray(images[i])
        image_io = io.BytesIO()
        pil_image.save(image_io, 'PNG')
        image_io.seek(0)
        images_dict[selected_image_id] = base64.b64encode(image_io.read()).decode('utf-8')

    return jsonify(images_dict)
