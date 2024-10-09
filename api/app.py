from flask import Flask, render_template, redirect, request, jsonify, current_app

import numpy as np
from PIL import Image
from api.projects.athlete_progan.eval import gen_images, upscale_images, TEAMS, TEAM_NAMES, BUILDS, SKIN_TONES
from api.init import init_generator, init_face_restorer
import io
import base64

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
    # Create zipped versions of property name and display name and send them to the HTML renderer
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
    if 'netG' not in current_app.config:
        # If the generator hasn't been loaded yet, then load it
        netG = init_generator('api/projects/athlete_progan/generator.pth')
        current_app.config['netG'] = netG
    else:
        netG = current_app.config['netG']

    # From the request form, receive the user's inputs
    team = request.form.get('team')
    skin_tone = request.form.get('skin-tone')
    build = request.form.get('build')
    selected_image_ids = request.form.getlist('ids[]')

    # Generate a batch of random images
    images = gen_images(netG, team, skin_tone, build, len(selected_image_ids))
    images = images.detach().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # Convert the generated images colour values from (-1, 1) to (0, 255)
    images = (images * 0.5 + 0.5) * 255
    images = images.astype(np.uint8)

    images_dict = {}
    # Iterate over every selected image id and assign it a randomly generated image
    for i, selected_image_id in enumerate(selected_image_ids):
        # Store the image for later access (e.g. upscaling that same image)
        current_app.config[selected_image_id] = images[i]

        pil_image = Image.fromarray(images[i])
        image_io = io.BytesIO()
        pil_image.save(image_io, 'PNG')
        image_io.seek(0)
        images_dict[selected_image_id] = base64.b64encode(image_io.read()).decode('utf-8')

    # Return the dictionary of images as a JSOn
    return jsonify(images_dict)

@app.route('/upscale', methods=['POST'])
def upscale():
    if 'face-restorer' not in current_app.config:
        # If the face restorer hasn't been loaded yet, then load it
        face_restorer = init_face_restorer()
        current_app.config['face-restorer'] = face_restorer
    else:
        face_restorer = current_app.config['face-restorer']

    # From the request form, receive the images the user selected
    selected_image_ids = request.form.getlist('ids[]')

    images_dict = {}
    # Iterate over every selected image id and upscale its image
    for i, selected_image_id in enumerate(selected_image_ids):
        if selected_image_id in current_app.config:
            np_image = current_app.config[selected_image_id]
            if np_image.shape == (117, 100, 3):
                # Only upscale the image if it exists and hasn't already been upscaled
                upscaled_np_image = face_restorer.process(np_image, aligned=False)[0]
                current_app.config[selected_image_id] = upscaled_np_image

                pil_image = Image.fromarray(upscaled_np_image)
                image_io = io.BytesIO()
                pil_image.save(image_io, 'PNG')
                image_io.seek(0)
                images_dict[selected_image_id] = base64.b64encode(image_io.read()).decode('utf-8')

    # Return the dictionary of images as a JSOn
    return jsonify(images_dict)
