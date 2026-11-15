"""MOMENT: frozen pretrained embeddings, 1,024 per task, 14,336 per person."""
DIMENSION_PER_TASK = 1024
DIMENSION_PER_PARTICIPANT = 14336


def extract(model, x):
    """Use the existing MOMENT API; caller owns inference mode and precision."""
    return model(x_enc=x).embeddings
