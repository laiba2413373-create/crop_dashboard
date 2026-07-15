from flask import (
    Flask,
    render_template,
    request
)

import os
import pandas as pd
import folium

from werkzeug.utils import secure_filename

from gee_backend import *

import ee
from flask import jsonify

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================
# PROVINCES & CROPS
# ==========================

PROVINCES = [
    "Punjab",
    "Sindh",
    "Balochistan",
    "Khyber Pakhtunkhwa"
]

# ==========================
# DISTRICTS
# ==========================

DISTRICTS = {}

CROPS = [
    "Wheat",
    "Rice",
    "Cotton",
    "Maize",
    "Sugarcane"
]


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

    return render_template(
        "index.html",
        provinces=PROVINCES,
        crops=CROPS,
        districts=[]
    )


# ==========================
# ABOUT PAGE
# ==========================

@app.route("/about")
def about():
    return render_template("about.html")

# ==========================
# GET DISTRICTS
# ==========================

@app.route("/districts/<province>")
def districts(province):

    district_list = get_districts(province)

    return jsonify(district_list)

# ==========================
# CLASSIFICATION
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    province = request.form["province"]
    district = request.form["district"]

    crop = request.form["crop"]

    model = request.form["model"]

    dataset_option = request.form["dataset"]

    region = get_region(
        province,
        district
    )
    start, end = get_season(crop)

    # ==========================
    # DEFAULT DATASET
    # ==========================

    if dataset_option == "default":

        train = get_training(
            table,
            province,
            district,
            crop
        )

    # ==========================
    # USER CSV
    # ==========================

    else:

        file = request.files["csv"]

        # File upload check
        if file.filename == "":

            return render_template(
                "index.html",
                error="Please upload a CSV file.",
                provinces=PROVINCES,
                crops=CROPS
            )

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
            district,
            crop
        )

        # No matching samples found
        if train is None:

            return render_template(
                "index.html",
                error=f"No {crop} samples found in {district}, {province}.",
                provinces=PROVINCES,
                crops=CROPS
            )

    # ==========================
    # MAP
    # ==========================

    centroid = region.centroid().coordinates().getInfo()

    m = folium.Map(

        location=[
            centroid[1],
            centroid[0]
        ],

        zoom_start=9,

        tiles=None

    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google Earth Engine / Google Maps",
        name="Google Earth Engine Map",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        attr="Google Earth Engine / Google Maps",
        name="Terrain Map",
        overlay=False,
        control=True,
        show=False
    ).add_to(m)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Earth Engine / Google Maps",
        name="RGB Map",
        overlay=False,
        control=True,
        show=False
    ).add_to(m)
    
    district_fc = ee.FeatureCollection(region)

    district_tile = district_fc.style(
        color="red",
        fillColor="00000000",
        width=3
    )

    tile = district_tile.getMapId({})

    folium.TileLayer(
        tiles=tile["tile_fetcher"].url_format,
        attr="Earth Engine",
        overlay=True,
        name="Selected District"
    ).add_to(m)

    rf_acc = None
    svm_acc = None
    rf_matrix = None
    svm_matrix = None

    # ==========================
    # RANDOM FOREST
    # ==========================

    if model in ["RF", "Both"]:

        rf_map, rf_acc, rf_matrix = run_rf(
            region,
            train,
            start,
            end
        )

        rf_tile = rf_map.getMapId({

            "min": 0,
            "max": 1,
            "palette": [
              "#ffffff",
              "#0066ff"
            ]

        })

        folium.TileLayer(
            tiles=rf_tile["tile_fetcher"].url_format,
            name="Random Forest",
            overlay=True,
            show=True,
            attr="Earth Engine",
            opacity=0.75
        ).add_to(m)

    # ==========================
    # SVM
    # ==========================

    if model in ["SVM", "Both"]:

        svm_map, svm_acc, svm_matrix = run_svm(
            region,
            train,
            start,
            end
        )

        svm_tile = svm_map.getMapId({

            "min": 0,
            "max": 1,
            "palette": [
              "#ffffff",
              "#00aa00"
            ]

        })

        folium.TileLayer(
            tiles=svm_tile["tile_fetcher"].url_format,
            name="SVM",
            overlay=True,
            show=True,
            attr="Earth Engine",
            opacity=0.75
        ).add_to(m)

    if model == "Both":
        folium.LayerControl(collapsed=False).add_to(m)

    map_html = m._repr_html_()

    return render_template(

        "result.html",

        province=province,

        district=district,

        crop=crop,

        rf_acc=rf_acc,

        svm_acc=svm_acc,

        rf_matrix=rf_matrix,

        svm_matrix=svm_matrix,

        map_html=map_html

    )
    
def get_boundary_layer(region):

    return region.getInfo()

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":

    app.run(
        debug=False
    )
