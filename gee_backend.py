import ee
import pandas as pd

KPK_GAUL_NAMES = [
    "Khyber Pakhtunkhwa",
    "North-West Frontier"
]

MAP_NDVI_THRESHOLD = 0.40
MAP_FALLBACK_NDVI_THRESHOLD = 0.25
MAP_MIN_CONNECTED_PIXELS = 8
RF_MAP_PROBABILITY_THRESHOLD = 0.35
RF_MAP_NUMBER_OF_TREES = 100

SERVICE_ACCOUNT = "crop-dashboard@bubbly-sentinel-486808-v7.iam.gserviceaccount.com"
KEY_FILE = "bubbly-sentinel-486808-v7-94f12f733330.json"

credentials = ee.ServiceAccountCredentials(
    SERVICE_ACCOUNT,
    KEY_FILE
)

ee.Initialize(
    credentials,
    project="bubbly-sentinel-486808-v7"
)

# ==================================
# DISTRICTS
# ==================================
# FILE: gee_backend.py

def get_districts(province):

    province_names = (
        KPK_GAUL_NAMES
        if province == "Khyber Pakhtunkhwa"
        else [province]
    )

    districts = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM0_NAME", "Pakistan"))
        .filter(ee.Filter.inList("ADM1_NAME", province_names))
        .aggregate_array("ADM2_NAME")
        .getInfo()
    )

    return sorted(list(set(districts)))

# ==================================
# DISTRICT REGION
# ==================================

def get_region(province, district):

    """
    Returns geometry of the selected district.
    """

    province_names = (
        KPK_GAUL_NAMES
        if province == "Khyber Pakhtunkhwa"
        else [province]
    )

    district_fc = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(
            ee.Filter.eq(
                "ADM0_NAME",
                "Pakistan"
            )
        )
        .filter(
            ee.Filter.inList(
                "ADM1_NAME",
                province_names
            )
        )
        .filter(
            ee.Filter.eq(
                "ADM2_NAME",
                district
            )
        )
    )

    if district_fc.size().getInfo() == 0:
        raise Exception(
            f"District '{district}' not found."
        )

    return district_fc.geometry()

def get_province_region(province):

    """
    Province geometry for training.
    """

    province_names = (
        KPK_GAUL_NAMES
        if province == "Khyber Pakhtunkhwa"
        else [province]
    )

    return (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter(
            ee.Filter.eq(
                "ADM0_NAME",
                "Pakistan"
            )
        )
        .filter(
            ee.Filter.inList(
                "ADM1_NAME",
                province_names
            )
        )
        .geometry()
    )


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
# DEFAULT TRAINING DATASET
# ==================================

def get_training(table, province, district, crop):

    """
    Province-wise training
    District preferred (if enough samples)
    Otherwise province fallback
    """

    province_region = get_province_region(province)

    data = table.filterBounds(province_region)

    print("All points:", data.size().getInfo())

    # -----------------------------
    # Crop filter
    # -----------------------------

    if crop == "Wheat":

        crop_points = data.filter(
            ee.Filter.eq("Description", "Wheat")
        )

    elif crop == "Rice":

        crop_points = data.filter(
            ee.Filter.eq("Description", "Rice")
        )

    elif crop == "Cotton":

        crop_points = data.filter(
            ee.Filter.eq("Code", 2)
        )

    elif crop == "Sugarcane":

        crop_points = data.filter(
            ee.Filter.eq("Code", 3)
        )

    elif crop == "Maize":

        crop_points = data.filter(
            ee.Filter.eq("Code", 7)
        )

    else:

        crop_points = ee.FeatureCollection([])

    # -----------------------------
    # Remove Orchard samples
    # -----------------------------

    crop_points = crop_points.filter(
        ee.Filter.neq("Land", "Orchard")
    )

    print("Crop points:", crop_points.size().getInfo())

    crop_points = (
        crop_points
        .randomColumn("rand")
        .sort("rand")
        .limit(400)
        .map(lambda f: f.set("class", 1))
    )

    # -----------------------------
    # Stronger background
    # -----------------------------

    background = (
        ee.FeatureCollection.randomPoints(
            region=province_region,
            points=800,
            seed=42
        )
        .map(lambda f: f.set("class", 0))
    )

    print("Crop Points:", crop_points.size().getInfo())
    print("Background:", background.size().getInfo())
    
    training = crop_points.merge(background)

    training = (
        training
        .randomColumn("rand")
        .sort("rand")
    )

    return training

# ==================================
# CSV TO EE
# ==================================

def csv_to_ee(df, province, district, crop):

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

    region = get_region(province, district)

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

# ==================================
# BUILD IMAGE (UPDATED)
# ==================================

def build_image(region, start, end):

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                20
            )
        )
    )

    image = ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),

            collection.median(),

            ee.Image.constant(
                [0,0,0,0,0,0]
            ).rename([
                "B2",
                "B3",
                "B4",
                "B8",
                "B11",
                "B12"
            ])
        )
    )

    image = image.select([
        "B2",
        "B3",
        "B4",
        "B8",
        "B11",
        "B12"
    ])

    ndvi = image.normalizedDifference(
        ["B8", "B4"]
    ).rename("NDVI")

    ndwi = image.normalizedDifference(
        ["B3", "B8"]
    ).rename("NDWI")

    evi = image.expression(

        "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",

        {

            "NIR": image.select("B8"),
            "RED": image.select("B4"),
            "BLUE": image.select("B2")

        }

    ).rename("EVI")

    savi = image.expression(

        "((NIR-RED)/(NIR+RED+0.5))*1.5",

        {

            "NIR": image.select("B8"),
            "RED": image.select("B4")

        }

    ).rename("SAVI")

    bsi = image.expression(

        "((SWIR+RED)-(NIR+BLUE))/((SWIR+RED)+(NIR+BLUE))",

        {

            "SWIR": image.select("B11"),
            "RED": image.select("B4"),
            "NIR": image.select("B8"),
            "BLUE": image.select("B2")

        }

    ).rename("BSI")

    image = image.addBands([
        ndvi,
        ndwi,
        evi,
        savi,
        bsi
    ])

    return image


# ==================================
# BANDS
# ==================================

def get_bands():

    return [

        "B2",
        "B3",
        "B4",
        "B8",
        "B11",
        "B12",

        "NDVI",
        "NDWI",
        "EVI",
        "SAVI",
        "BSI"

    ]
    
    
def get_svm_bands():

    return [

        "B2",

        "B3",

        "B4",

        "B8",

        "B11",

        "NDVI",

        "EVI"

    ]


# ==================================
# MAP OUTPUT
# ==================================

def finalize_classification(classified, ndvi, cropland, region):

    classified = classified.eq(1)

    # Keep only cropland
    classified = classified.updateMask(cropland)

    # Remove very low vegetation
    classified = classified.updateMask(
        ndvi.gt(0.30)
    )

    # Remove tiny isolated pixels
    connected = classified.connectedPixelCount(
        maxSize=50,
        eightConnected=True
    )

    classified = classified.updateMask(
        connected.gte(3)
    )

    return (
        classified
        .selfMask()
        .rename("classification")
        .toByte()
        .clip(region)
    )


# ==================================
# RANDOM FOREST
# ==================================

def run_rf(region, train, start, end):

    bands = get_bands()

    img = build_image(
        region,
        start,
        end
    )

    if img.bandNames().size().getInfo() == 0:

        raise Exception(
            "No Sentinel-2 imagery available."
        )

    # ===========================
    # WorldCover Cropland Mask
    # ===========================

    cropland = (
        ee.Image("ESA/WorldCover/v200/2021")
        .eq(40)
    )

    ndvi = img.select("NDVI")

    # ===========================
    # Sample Training
    # ===========================

    samples = img.sampleRegions(
        collection=train,
        properties=["class"],
        scale=10,
        geometries=False,
        tileScale=4
    )

    samples = (
        samples
        .randomColumn("rand")
        .sort("rand")
        .limit(1200)
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

    # ===========================
    # Random Forest
    # ===========================

    rf_classifier = ee.Classifier.smileRandomForest(

        numberOfTrees=300,
        variablesPerSplit=4,
        minLeafPopulation=3,
        bagFraction=0.8,
        seed=42

    )

    model = rf_classifier.train(

        features=train_data,
        classProperty="class",
        inputProperties=bands

    )

    map_classifier = ee.Classifier.smileRandomForest(

        numberOfTrees=RF_MAP_NUMBER_OF_TREES,
        variablesPerSplit=4,
        minLeafPopulation=3,
        bagFraction=0.8,
        seed=42

    ).setOutputMode("PROBABILITY")

    map_model = map_classifier.train(

        features=train_data,
        classProperty="class",
        inputProperties=bands

    )

    # ===========================
    # Accuracy
    # ===========================

    classified_test = test_data.classify(model)

    error_matrix = classified_test.errorMatrix(
        "class",
        "classification"
    )

    accuracy = error_matrix.accuracy().getInfo()
    confusion_matrix = error_matrix.array().getInfo()

    if accuracy <= 1:
        accuracy *= 100

    print("RF Accuracy:", accuracy)

    # ===========================
    # Prediction
    # ===========================

    classified = finalize_classification(
        img.classify(map_model).gte(RF_MAP_PROBABILITY_THRESHOLD),
        ndvi,
        cropland,
        region
    )

    return classified, round(accuracy, 2), confusion_matrix

# ==================================
# SVM
# ==================================
def run_svm(region, train, start, end):

    bands = get_svm_bands()

    img = build_image(
        region,
        start,
        end
    )

    if img.bandNames().size().getInfo() == 0:
        raise Exception(
            "No Sentinel-2 imagery available."
        )

    # ==================================
    # Cropland Mask
    # ==================================

    cropland = (
        ee.Image("ESA/WorldCover/v200/2021")
        .eq(40)
    )

    ndvi = img.select("NDVI")

    # ==================================
    # Min-Max Feature Scaling
    # ==================================

    stats = img.select(bands).reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=region,
        scale=30,
        bestEffort=True,
        maxPixels=1e9,
        tileScale=4
    )

    scaled_bands = []

    for b in bands:

        b_min = ee.Number(stats.get(b + "_min"))
        b_max = ee.Number(stats.get(b + "_max"))
        b_range = b_max.subtract(b_min).max(1e-6)

        scaled = (
            img.select(b)
            .subtract(b_min)
            .divide(b_range)
            .rename(b)
        )

        scaled_bands.append(scaled)

    img_scaled = ee.Image.cat(scaled_bands)

    # ==================================
    # Sample Training Data
    # ==================================

    samples = img_scaled.sampleRegions(
        collection=train,
        properties=["class"],
        scale=10,
        tileScale=4
    )

    samples = (
        samples
        .randomColumn("rand")
        .sort("rand")
        .limit(1200)
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

    # ==================================
    # SVM
    # ==================================

    model = ee.Classifier.libsvm(
        kernelType="LINEAR"

    ).train(
        features=train_data,
        classProperty="class",
        inputProperties=bands
    )

    # ==================================
    # Accuracy
    # ==================================

    classified_test = test_data.classify(model)

    error_matrix = classified_test.errorMatrix(
        "class",
        "classification"
    )

    accuracy = error_matrix.accuracy().getInfo()
    confusion_matrix = error_matrix.array().getInfo()

    if accuracy <= 1:
        accuracy *= 100

    print("SVM Accuracy:", accuracy)

    # ==================================
    # Prediction
    # ==================================

    classified = img_scaled.classify(model)

    classified = classified.eq(1)

    classified = classified.selfMask()

    classified = classified.clip(region)

    return classified, round(accuracy, 2), confusion_matrix


























