import io
import uuid
import base64

from flask_session import Session
from flask import Flask, request, render_template, send_file, session

from src.avatar_generator import AvatarGenerator
from src.avatar_generator import form_caption, pil_2_bytes
from flask_app.secrets.session_secret import SECRET_KEY

avatar_generator = AvatarGenerator()

app = Flask(__name__)
SESSION_TYPE = "filesystem"
app.secret_key = SECRET_KEY
app.config.from_object(__name__)
Session(app)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        avatar_config = request.form.to_dict()
        caption = form_caption(avatar_config)
        generated_avatar = avatar_generator.generate_avatars(caption)[0]
        contents = pil_2_bytes(generated_avatar)
        session["contents"] = contents
        encoded_image_data = base64.b64encode(contents)
        return render_template("result_page.html", generated_avatar=encoded_image_data.decode('utf-8'))
    else:
        session.clear()
        return render_template("main.html")


@app.route("/download")
def download():
    send_file(io.BytesIO(session["contents"]), mimetype="image/png",
              download_name=str(uuid.uuid4()), as_attachment=True)
    return send_file(io.BytesIO(session["contents"]), mimetype="image/png",
                     download_name=str(uuid.uuid4()), as_attachment=True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4567)
