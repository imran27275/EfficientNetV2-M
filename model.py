import torch
import timm

class EfficientNetModel:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading EfficientNetV2-M model...")

        # for ConvNeXt-B model use this code and remove the self.model code below
        # self.model = timm.create_model(
        #     "convnext_base",
        #     pretrained=True,
        #     num_classes=2
        # )

        self.model = timm.create_model(
            "tf_efficientnetv2_m",
            pretrained=True,
            num_classes=2
        )
        self.model = self.model.to(self.device)
        print("Model loaded successfully")