# since I'll be using the pretrained models a lot in other files, I'll put them here to be imported
PRETRAINED_MODELS: dict[str, str] = {
    "starry_night": "saved_models/starry_night_pretrained.pth",
    "rain_princess": "saved_models/rain_princess_pretrained.pth",
    "abstract": "saved_models/abstract_pretrained.pth",
    "mosaic": "saved_models/mosaic_pretrained.pth",
    "cht1": "saved_models/cht1_pretrained.pth",
    "cht2": "saved_models/cht2_pretrained.pth",
    "cht3": "saved_models/cht3_pretrained.pth",
    "cht4": "saved_models/cht3_3loss.pth",
    "cht5": "saved_models/cht3_2loss.pth",
    "cht6": "saved_models/cht3_white_26.pth",
    "cht7": "saved_models/ch2_white_23.pth",
    "cht0": "saved_models/test.pth",
    "cht8": "saved_models/test2.pth",
    "cht9": "saved_models/aba_no_lap.pth",
    "cht10": "saved_models/aba_no_lap2.pth",
    "cht11": "saved_models/aba_no_lap3.pth",
    "cht12": "saved_models/final023.pth",
    "cht13": "saved_models/only_lap023_0_1.pth",
    "default": "saved_models/trained_model.pth",
} # I delete or rename most of these.

# TODO: retrain the abstract model with slightly different image and hyperparameters
