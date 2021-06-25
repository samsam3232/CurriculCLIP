ResNet50_params = {
    "vision_layers": (3, 4, 6, 3),
    "vision_width": 64,
    "image_resolution": 224,
    "embed_dim": 1024,
    "context_length": 77,
    "vocab_size": 49408,
    "transformer_width": 512,
    "transformer_heads": 8,
    "transformer_layers": 12,
    "vision_patch_size": None
}

params = {"ResNet50": ResNet50_params}