from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load translation model once
translation = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="eng_Latn",
    tgt_lang="tam_Taml"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "Please enter some text."
        }), 400

    try:
        result = translation(text)

        translated_text = result[0]["translation_text"]

        return jsonify({
            "translation": translated_text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)