# import gradio as gr
# import ee
# from gee_backend import *

# # -----------------------------
# # INIT EARTH ENGINE
# # -----------------------------
# SERVICE_ACCOUNT = "crop-dashboard@bubbly-sentinel-486808-v7.iam.gserviceaccount.com"
# KEY_FILE = "bubbly-sentinel-486808-v7-94f12f733330.json"

# credentials = ee.ServiceAccountCredentials(
#     SERVICE_ACCOUNT,
#     KEY_FILE
# )

# ee.Initialize(credentials, project="bubbly-sentinel-486808-v7")


# # -----------------------------
# # FUNCTION
# # -----------------------------
# def predict(province, crop, model):

#     region = get_region(province)
#     start, end = get_season(crop)

#     table = ee.FeatureCollection(
#         "projects/bubbly-sentinel-486808-v7/assets/Pakistan_Agricultural"
#     )

#     train = get_training(table, region, crop)

#     rf_acc = None
#     svm_acc = None

#     if model in ["RF", "Both"]:
#         _, rf_acc = run_rf(region, train, start, end)

#     if model in ["SVM", "Both"]:
#         _, svm_acc = run_svm(region, train, start, end)

#     return f"RF Accuracy: {rf_acc} | SVM Accuracy: {svm_acc}"


# # -----------------------------
# # UI
# # -----------------------------
# interface = gr.Interface(
#     fn=predict,
#     inputs=[
#         gr.Dropdown(["Punjab","Sindh","Balochistan","Khyber Pakhtunkhwa"], label="Province"),
#         gr.Dropdown(["Wheat","Rice","Cotton","Maize","Sugarcane"], label="Crop"),
#         gr.Dropdown(["RF","SVM","Both"], label="Model")
#     ],
#     outputs="text",
#     title="🌾 Crop Classification System"
# )

# interface.launch()