import yaml


def load_config(is_prod: bool = False) -> dict:
    """
    Carga un archivo de configuración YAML.

    Args:
        config_path (str): Ruta al archivo de configuración YAML

    Returns:
        dict: Contenido del archivo YAML como diccionario
    """

    if is_prod:
        config_path = "conf/config_prod.yaml"
    else:
        config_path = "conf/config_dev.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
