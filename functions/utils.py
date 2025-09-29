import yaml


def load_config(config_path: str) -> dict:
    """
    Carga un archivo de configuración YAML.

    Args:
        config_path (str): Ruta al archivo de configuración YAML

    Returns:
        dict: Contenido del archivo YAML como diccionario
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
