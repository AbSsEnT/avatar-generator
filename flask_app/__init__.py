import base64
import io
import uuid
from flask import Flask, request, render_template, send_file, session

from src.avatar_generator import AvatarGenerator
from src.avatar_generator import form_caption, pil_2_bytes
from flask_app.secrets.session_secret import SECRET_KEY

avatar_generator = AvatarGenerator()
app = Flask(__name__, instance_relative_config=True, template_folder="templates")
app.secret_key = SECRET_KEY


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        avatar_config = request.form.to_dict()
        caption = form_caption(avatar_config)
        generated_avatar = avatar_generator.generate_avatars(caption, n_samples=1)[0]
        contents = pil_2_bytes(generated_avatar)
        session["contents"] = contents
        encoded_image_data = base64.b64encode(contents)
        return render_template("result_page.html", generated_avatar=encoded_image_data.decode('utf-8'))
    else:
        session.clear()
        return render_template("main.html")


@app.route("/download")
def download():
    return send_file(io.BytesIO(session["contents"]), mimetype="image/png",
                     download_name=str(uuid.uuid4()), as_attachment=True)
