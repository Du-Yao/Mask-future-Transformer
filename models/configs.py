model_config = {
    # "num_class, dim, depth, heads, dim_head, mlp_dim, dropout = 0., pool='gap'"
    "num_class": 100,
    "dim": 768,
    "depth": 8,
    "heads": 12,
    "dim_head": 64,
    "mlp_dim": 2048,
    "dropout": 0.1,
    "pool": "gap"
}

vit_config = {
        # image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls'
        "image_size": 224,
        "patch_size": 16,
        "num_classes": 100,
        "dim": 768,
        "depth": 8,
        "heads": 12,
        "mlp_dim": 2048,
        "pool": "cls"
    }