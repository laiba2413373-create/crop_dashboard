from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for
)

import os
import pandas as pd
import folium

from werkzeug.utils import secure_filename

from gee_backend import *

import ee

ee.Initialize(
    project="bubbly-sentinel-486808-v7"
)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================
# DEFAULT DATASET
# ==========================

table = ee.FeatureCollection(
    "projects/bubbly-sentinel-486808-v7/assets/Pakistan_Agricultural"
)


# ==========================
# HOME PAGE
# ==========================

@app.route("/")
def index():

    provinces = [
        "Punjab",
        "Sindh",
        "Balochistan",
        "Khyber Pakhtunkhwa"
    ]

    crops = [
        "Wheat",
        "Rice",
        "Cotton",
        "Maize",
        "Sugarcane"
    ]

    return render_template(
        "index.html",
        provinces=provinces,
        crops=crops
    )


# ==========================
# ABOUT PAGE
# ==========================

@app.route("/about")
def about():
    return render_template("about.html")


# ==========================
# CLASSIFICATION
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    province = request.form["province"]
    crop = request.form["crop"]
    model = request.form["model"]

    dataset_option = request.form["dataset"]

    region = get_region(province)

    start, end = get_season(crop)

    # ==========================
    # DEFAULT DATASET
    # ==========================

    if dataset_option == "default":

        train = get_training(
            table,
            region,
            crop
        )

    # ==========================
    # USER CSV
    # ==========================

    else:

        file = request.files["csv"]

        filename = secure_filename(file.filename)

        path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(path)

        df = pd.read_csv(path)

        train = csv_to_ee(
            df,
            province,
            crop
        )
        
        if train is None:

            return render_template(
                "index.html",
                error="No points found for selected crop and province."
            )

    # ==========================
    # MAP
    # ==========================

    m = folium.Map(
        location=[30.5, 69.3],
        zoom_start=6,
        tiles="CartoDB dark_matter"
    )

    rf_acc = None
    svm_acc = None

    # ==========================
    # RANDOM FOREST
    # ==========================

    if model in ["RF", "Both"]:

        rf_map, rf_acc = run_rf(
            region,
            train,
            start,
            end
        )

        rf_tile = rf_map.getMapId({
            "min": 0,
            "max": 1,
            "palette": [
                "000000",
                "00ff00"
            ]
        })

        folium.TileLayer(
            tiles=rf_tile["tile_fetcher"].url_format,
            name="Random Forest",
            overlay=True,
            attr="Earth Engine"
        ).add_to(m)

    # ==========================
    # SVM
    # ==========================

    if model in ["SVM", "Both"]:

        svm_map, svm_acc = run_svm(
            region,
            train,
            start,
            end
        )

        svm_tile = svm_map.getMapId({
            "min": 0,
            "max": 1,
            "palette": [
                "000000",
                "00aaff"
            ]
        })

        folium.TileLayer(
            tiles=svm_tile["tile_fetcher"].url_format,
            name="SVM",
            overlay=True,
            attr="Earth Engine"
        ).add_to(m)

    folium.LayerControl().add_to(m)

    map_html = m._repr_html_()

    return render_template(
        "result.html",
        province=province,
        crop=crop,
        rf_acc=rf_acc,
        svm_acc=svm_acc,
        map_html=map_html
    )


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":

    app.run(
        debug=True
    )
    
