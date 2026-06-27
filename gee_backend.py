import ee
import pandas as pd
import os
import json
import tempfile

SERVICE_ACCOUNT = "crop-dashboard@bubbly-sentinel-486808-v7.iam.gserviceaccount.com"

if os.path.exists("bubbly-sentinel-486808-v7-94f12f733330.json"):

    credentials = ee.ServiceAccountCredentials(
        SERVICE_ACCOUNT,
        "bubbly-sentinel-486808-v7-94f12f733330.json"
    )
else:

    key_data = json.loads(os.environ["EE_KEY"])

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=".json"
    )

    json.dump(key_data, temp_file)
    temp_file.close()

    credentials = ee.ServiceAccountCredentials(
        SERVICE_ACCOUNT,
        temp_file.name
    )

ee.Initialize(
    credentials,
    project="bubbly-sentinel-486808-v7"
)



# ==================================
# REGION
# ==================================

def get_region(province):

    if province == "Khyber Pakhtunkhwa":
        province = "North-West Frontier"

    region = (
        ee.FeatureCollection(
            "FAO/GAUL/2015/level1"
        )
        .filter(
            ee.Filter.eq(
                "ADM0_NAME",
                "Pakistan"
            )
        )
        .filter(
            ee.Filter.eq(
                "ADM1_NAME",
                province
            )
        )
        .geometry()
    )

    return region


# ==================================
# SEASONS
# ==================================

def get_season(crop):

    seasons = {

        "Wheat":
        ("2023-11-01", "2024-03-31"),

        "Rice":
        ("2023-07-01", "2023-10-31"),

        "Cotton":
        ("2023-05-01", "2023-10-31"),

        "Maize":
        ("2023-06-01", "2023-09-30"),

        "Sugarcane":
        ("2023-04-01", "2023-12-31")
    }

    return seasons[crop]


# ==================================
# DEFAULT DATASET
# ==================================

def get_training(table, region, crop):

    data = table.filterBounds(region)

    if crop == "Wheat":

        crop_points = data.filter(
            ee.Filter.eq(
                "Description",
                "Wheat"
            )
        )

    elif crop == "Rice":

        crop_points = data.filter(
            ee.Filter.eq(
                "Description",
                "Rice"
            )
        )

    elif crop == "Cotton":

        crop_points = data.filter(
            ee.Filter.eq(
                "Code",
                2
            )
        )

    elif crop == "Sugarcane":

        crop_points = data.filter(
            ee.Filter.eq(
                "Code",
                3
            )
        )

    elif crop == "Maize":

        crop_points = data.filter(
            ee.Filter.eq(
                "Code",
                7
            )
        )

    crop_points = (
        crop_points
        .limit(500)
        .map(
            lambda f:
            f.set("class", 1)
        )
    )

    background = (
        ee.FeatureCollection.randomPoints(
            region=region,
            points=300,
            seed=1
        )
        .map(
            lambda f:
            f.set("class", 0)
        )
    )

    return crop_points.merge(background)


# ==================================
# CSV TO EE
# ==================================

def csv_to_ee(df, province, crop):

    df.columns = df.columns.str.lower()

    df["province"] = (
        df["province"]
        .str.strip()
    )

    df["crop"] = (
        df["crop"]
        .str.strip()
    )

    df = df[
        (df["province"] == province)
        &
        (df["crop"] == crop)
    ]

    if len(df) == 0:
        return None

    features = []

    for _, row in df.iterrows():

        point = ee.Feature(
            ee.Geometry.Point([
                row["longitude"],
                row["latitude"]
            ]),
            {"class": 1}
        )

        features.append(point)

    region = get_region(province)

    background = (
        ee.FeatureCollection.randomPoints(
            region=region,
            points=300,
            seed=1
        )
        .map(
            lambda f:
            f.set("class", 0)
        )
    )

    return (
        ee.FeatureCollection(features)
        .merge(background)
    )


# ==================================
# IMAGE
# ==================================

def build_image(region, start, end):

    bands = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B11"
    ]

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(region)
        .filterDate(start, end)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                30
            )
        )
    )

    image = ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),

            collection.median(),

            ee.Image.constant(
                [0, 0, 0, 0, 0]
            ).rename(bands)
        )
    )

    return image.select(bands)


# ==================================
# RANDOM FOREST
# ==================================

def run_rf(region, train, start, end):

    bands = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B11"
    ]

    img = build_image(
        region,
        start,
        end
    )

    samples = img.sampleRegions(
        collection=train,
        properties=["class"],
        scale=10,
        geometries=True
    )

    count = samples.size().getInfo()

    print("RF Samples:", count)

    if count < 20:
        raise Exception(
            f"Only {count} training samples found."
        )

    samples = samples.randomColumn("random")

    train_data = samples.filter(
        ee.Filter.lt("random", 0.7)
    )

    test_data = samples.filter(
        ee.Filter.gte("random", 0.7)
    )

    model = ee.Classifier.smileRandomForest(
        numberOfTrees=50
    ).train(
        features=train_data,
        classProperty="class",
        inputProperties=bands
    )

    classified_test = test_data.classify(model)

    accuracy = (
        classified_test
        .errorMatrix(
            "class",
            "classification"
        )
        .accuracy()
        .getInfo()
    )

    classified = (
        img
        .classify(model)
        .clip(region)
    )

    if accuracy <= 1:
        accuracy = accuracy * 100
    print("RF Accuracy:", accuracy)
    return classified, round(accuracy, 2)


# ==================================
# SVM
# ==================================

def run_svm(region, train, start, end):

    bands = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B11"
    ]

    img = build_image(
        region,
        start,
        end
    )

    samples = img.sampleRegions(
        collection=train,
        properties=["class"],
        scale=10,
        geometries=True
    )

    count = samples.size().getInfo()

    print("SVM Samples:", count)

    if count < 20:
        raise Exception(
            f"Only {count} training samples found."
        )

    samples = samples.randomColumn("random")

    train_data = samples.filter(
        ee.Filter.lt("random", 0.7)
    )

    test_data = samples.filter(
        ee.Filter.gte("random", 0.7)
    )

    model = ee.Classifier.libsvm(
        kernelType="RBF",
        gamma=0.1,
        cost=1
    ).train(
        features=train_data,
        classProperty="class",
        inputProperties=bands
    )

    classified_test = test_data.classify(model)

    accuracy = (
        classified_test
        .errorMatrix(
            "class",
            "classification"
        )
        .accuracy()
        .getInfo()
    )

    classified = (
        img
        .classify(model)
        .clip(region)
    )

    if accuracy <= 1:
        accuracy = accuracy * 100
    
    print("SVM Accuracy:", accuracy)
    return classified, round(accuracy, 2)
