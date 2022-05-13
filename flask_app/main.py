import io
import uuid
from flask import Flask, request, render_template, redirect, url_for, send_file

from src.avatar_generator import AvatarGenerator
from src.avatar_generator import form_caption, pil_2_bytes

avatar_generator = AvatarGenerator()
app = Flask(__name__)


# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         avatar_config = request.form.to_dict()
#         caption = form_caption(avatar_config)
#         generated_avatars = avatar_generator.generate_avatars(caption, n_samples=1)  # ByteString.
#         return redirect(url_for("result", generated_avatars=generated_avatars))
#     else:
#         return render_template("index.html")
#
#
# @app.route("/result")
# def result():
#     return render_template("result.html", generated_avatars=request.args.get("generated_avatars"))


@app.route("/")
def index():
    caption = "dummy_caption"
    generated_avatar = avatar_generator.generate_avatars(caption, n_samples=1)[0]
    contents = pil_2_bytes(generated_avatar)
    return send_file(io.BytesIO(contents), mimetype="image/png", download_name=str(uuid.uuid4()), as_attachment=True)
