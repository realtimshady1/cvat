# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = "/opt/nuclio/sam/sam_hq_vit_h.pth"
        self.model_type = "vit_h"
        self.latest_image = None
        sam_model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam_model.to(device=self.device)
        self.predictor = SamPredictor(sam_model)

    def handle(self, image):
        self.predictor.set_image(np.array(image))
        features = self.predictor.get_image_embedding()
        interm_embeddings = torch.stack(self.predictor.interm_features)
        return features, interm_embeddings