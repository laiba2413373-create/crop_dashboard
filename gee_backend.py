import os
import json
import tempfile
import ee

SERVICE_ACCOUNT = os.environ["EE_SERVICE_ACCOUNT"]

service_account_json = json.loads(
    os.environ["EE_KEY"]
)

temp_key = tempfile.NamedTemporaryFile(
    mode="w",
    suffix=".json",
    delete=False
)

json.dump(service_account_json, temp_key)
temp_key.close()

credentials = ee.ServiceAccountCredentials(
    SERVICE_ACCOUNT,
    temp_key.name
)

ee.Initialize(
    credentials,
    project="bubbly-sentinel-486808-v7"
)

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

    province_region = get_province_region(province)
    district_region = get_region(province, district)

    district_data = table.filterBounds(district_region)
    province_data = table.filterBounds(province_region)

    def pure_crop(fc, crop_name):

        if crop_name == "Wheat":
            return fc.filter(ee.Filter.eq("Description", "Wheat"))

        elif crop_name == "Rice":
            return fc.filter(ee.Filter.eq("Description", "Rice"))

        elif crop_name == "Cotton":
            return fc.filter(ee.Filter.eq("Code", 2))

        elif crop_name == "Sugarcane":
            return fc.filter(ee.Filter.eq("Code", 3))

        elif crop_name == "Maize":
            return fc.filter(ee.Filter.eq("Code", 7))

        return ee.FeatureCollection([])

    district_positive = pure_crop(district_data, crop)
    district_count = district_positive.size().getInfo()

    print("District crop samples :", district_count)

    if district_count >= 150:
        print("Using District Training")
        data = district_data
    else:
        print("Using Province Training")
        data = province_data

    positives = (
        pure_crop(data, crop)
        .randomColumn("rand")
        .sort("rand")
        .limit(400)
        .map(lambda f: f.set("class", 1))
    )

    positive_count = positives.size().getInfo()

    if positive_count == 0:
        raise Exception(
            f"No '{crop}' training samples found for {district}. Please choose another crop or district."
        )

    if positive_count < 30:
        raise Exception(
            f"Only {positive_count} '{crop}' training samples found. At least 30 samples are required."
        )

    crop_names = [
        "Wheat",
        "Rice",
        "Cotton",
        "Sugarcane",
        "Maize"
    ]

    negatives = ee.FeatureCollection([])

    for c in crop_names:

        if c != crop:
            negatives = negatives.merge(
                pure_crop(data, c)
            )

    non_crop = data.filter(

        ee.Filter.inList(

            "Land",

            [
                "Barrenland",
                "Builtup",
                "Water",
                "Forest",
                "Grassland",
                "Shrub",
                "Shrubs",
                "Fallowland"
            ]

        )

    )

    negatives = (
        negatives
        .merge(non_crop)
        .randomColumn("rand")
        .sort("rand")
        .limit(400)
        .map(lambda f: f.set("class",0))
    )

    negative_count = negatives.size().getInfo()

    if negative_count < 30:
        raise Exception(
            "Not enough background samples available for training."
        )

    training = positives.merge(negatives)

    print("--------------------------")
    print("District :", district)
    print("Crop :", crop)
    print("Positive :", positive_count)
    print("Negative :", negative_count)
    print("Training :", training.size().getInfo())
    print("--------------------------")

    return training

# ==================================
# CSV TO EE
# ==================================

def csv_to_ee(df, province, district, crop):

    required_columns = [
        "latitude",
        "longitude",
        "province",
        "crop"
    ]

    df.columns = df.columns.str.lower().str.strip()

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise Exception(
            "Invalid CSV format. Required columns: latitude, longitude, province, crop."
        )

    if len(df) == 0:
        raise Exception("Uploaded CSV file is empty.")

    df["province"] = df["province"].astype(str).str.strip()
    df["crop"] = df["crop"].astype(str).str.strip()

    df = df[
        (df["province"] == province)
        &
        (df["crop"] == crop)
    ]

    if len(df) == 0:
        raise Exception(
            f"No '{crop}' points found for province '{province}' in uploaded CSV."
        )

    if len(df) < 20:
        raise Exception(
            f"Only {len(df)} crop points found. At least 20 points are required."
        )

    features = []

    for _, row in df.iterrows():

        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue

        features.append(

            ee.Feature(

                ee.Geometry.Point([
                    row["longitude"],
                    row["latitude"]
                ]),

                {"class":1}

            )

        )

    if len(features) == 0:
        raise Exception(
            "Uploaded CSV does not contain valid coordinates."
        )

    region = get_region(province, district)

    background = (
        ee.FeatureCollection.randomPoints(
            region=region,
            points=300,
            seed=1
        )
        .map(lambda f: f.set("class",0))
    )

    return ee.FeatureCollection(features).merge(background)

# ==================================
# BUILD IMAGE
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

    image = collection.median()

    image = image.select([
        "B2",
        "B3",
        "B4",
        "B8",
        "B11",
        "B12"
    ]).divide(10000)

    # -----------------------------
    # Vegetation Indices
    # -----------------------------

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

        "1.5*((NIR-RED)/(NIR+RED+0.5))",

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


# ==================================
# MAP OUTPUT
# ==================================

def finalize_classification(classified, ndvi, cropland, region):

    return (
        classified.eq(1)
        .selfMask()
        .updateMask(ndvi.gt(MAP_FALLBACK_NDVI_THRESHOLD))
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

    # ---------------------------------
    # Sample Training Data
    # ---------------------------------

    samples = img.sampleRegions(

        collection=train,

        properties=["class"],

        scale=10,

        geometries=False,

        tileScale=4

    )

    count = samples.size().getInfo()

    print("RF Samples :", count)

    if count < 50:

        raise Exception(
            f"Only {count} valid training samples found. Atleast 50 are required."
        )
        
    # ---------------------------------
    # Train/Test Split
    # ---------------------------------

    samples = samples.randomColumn("split")

    train_data = samples.filter(
        ee.Filter.lt("split", 0.7)
    )

    test_data = samples.filter(
        ee.Filter.gte("split", 0.7)
    )
    
    if train_data.size().getInfo() < 20:
        raise Exception(
            "Random Forest training failed because too few training samples were available."
        )
    
    if test_data.size().getInfo() < 10:
        raise Exception(
            "Random Forest validation failed because too few testing samples were available."
        )

    # ---------------------------------
    # Random Forest
    # ---------------------------------

    classifier = ee.Classifier.smileRandomForest(

        numberOfTrees=300,
        variablesPerSplit=5,
        minLeafPopulation=3,
        bagFraction=0.8,
        seed=42

    ).train(

        features=train_data,

        classProperty="class",

        inputProperties=bands

    )

    # ---------------------------------
    # Accuracy
    # ---------------------------------

    test = test_data.classify(classifier)

    matrix = test.errorMatrix(
        "class",
        "classification"
    )

    accuracy = matrix.accuracy().getInfo()

    if accuracy <= 1:
        accuracy *= 100

    print("RF Accuracy :", accuracy)
    confusion_matrix = matrix.array().getInfo()

    print(confusion_matrix)
    
    if accuracy < 40:
        raise Exception(
            "Random Forest model accuracy is too low. Please use more training samples or another district."
        )

    # ---------------------------------
    # Classification
    # ---------------------------------

    classified = img.classify(classifier)

    classified = classified.eq(1)

    # ---------------------------------
    # Restrict to Cropland
    # ---------------------------------

    cropland = ee.Image(
        "ESA/WorldCover/v200/2021"
    ).eq(40)

    classified = classified.updateMask(
        cropland
    )

    # ---------------------------------
    # Healthy Vegetation Mask
    # ---------------------------------

    vegetation = (

        img.select("NDVI").gt(0.35)

        .And(
            img.select("EVI").gt(0.18)
        )

        .And(
            img.select("BSI").lt(0.15)
        )

    )

    classified = classified.updateMask(
        vegetation
    )

    # ---------------------------------
    # Remove Noise
    # ---------------------------------

    classified = classified.focal_mode(
        radius=1,
        units="pixels"
    )

    connected = classified.connectedPixelCount(
        maxSize=50,
        eightConnected=True
    )

    classified = classified.updateMask(
        connected.gte(8)
    )

    classified = classified.selfMask()

    classified = classified.clip(region)

    return (
        classified,
        round(accuracy, 2),
        matrix.getInfo()
    )

# ==================================
# SVM
# ==================================


def run_svm(region, train, start, end):

    bands = get_bands()

    img = build_image(
        region,
        start,
        end
    )

    # ---------------------------------
    # Sample Training Data
    # ---------------------------------

    samples = img.sampleRegions(

        collection=train,

        properties=["class"],

        scale=10,

        geometries=False,

        tileScale=4

    )

    count = samples.size().getInfo()

    print("SVM Samples :", count)

    if count < 50:

        raise Exception(
            f"Only {count} valid training samples found. Atleast 50 are required"
        )

    # ---------------------------------
    # Train/Test Split
    # ---------------------------------

    samples = samples.randomColumn("split")

    train_data = samples.filter(
        ee.Filter.lt("split", 0.7)
    )

    test_data = samples.filter(
        ee.Filter.gte("split", 0.7)
    )
    
    if train_data.size().getInfo() < 20:
        raise Exception(
            "SVM training failed because too few training samples were available."
        )
        
    if test_data.size().getInfo() < 10:
        raise Exception(
            "SVM validation failed because too few testing samples were available."
        )

    # ---------------------------------
    # SVM
    # ---------------------------------

    classifier = ee.Classifier.libsvm(

        kernelType="RBF",

        gamma=0.2,

        cost=100

    ).train(

        features=train_data,

        classProperty="class",

        inputProperties=bands

    )

    # ---------------------------------
    # Accuracy
    # ---------------------------------

    test = test_data.classify(classifier)

    matrix = test.errorMatrix(
        "class",
        "classification"
    )

    accuracy = matrix.accuracy().getInfo()

    if accuracy <= 1:
        accuracy *= 100

    print("SVM Accuracy :", accuracy)
    print(matrix.getInfo())
    
    if accuracy < 40:
        raise Exception(
            "SVM model accuracy is too low. Please use more training samples or another district."
        )

    # ---------------------------------
    # Classification
    # ---------------------------------

    classified = img.classify(classifier)

    classified = classified.eq(1)

    # ---------------------------------
    # Cropland Mask
    # ---------------------------------

    cropland = ee.Image(
        "ESA/WorldCover/v200/2021"
    ).eq(40)

    classified = classified.updateMask(
        cropland
    )

    # ---------------------------------
    # Vegetation Mask
    # ---------------------------------

    vegetation = (

        img.select("NDVI").gt(0.35)

        .And(
            img.select("EVI").gt(0.18)
        )

        .And(
            img.select("BSI").lt(0.15)
        )

    )

    classified = classified.updateMask(
        vegetation
    )

    # ---------------------------------
    # Remove Noise
    # ---------------------------------

    classified = classified.focal_mode(
        radius=1,
        units="pixels"
    )

    connected = classified.connectedPixelCount(
        maxSize=50,
        eightConnected=True
    )

    classified = classified.updateMask(
        connected.gte(8)
    )

    classified = classified.selfMask()

    classified = classified.clip(region)

    return (
        classified,
        round(accuracy, 2),
        matrix.getInfo()
    )
