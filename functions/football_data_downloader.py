from datetime import datetime
from pathlib import Path

import requests
from loguru import logger

from functions.utils import load_config

config = load_config()

# Mapeo de ligas principales
LEAGUE_CODES = config["configuration"]["league_codes"]


class FootballDataDownloader:
    """
    Clase para descargar datos de football-data.co.uk
    """

    def __init__(self, base_url="https://www.football-data.co.uk/mmz4281"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def get_current_season(self):
        """
        Determina la temporada actual basada en la fecha.
        La temporada de fútbol típicamente va de agosto a mayo.

        Returns:
            str: Código de temporada (ej: '2425' para 2024-2025)
        """
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month

        # Si estamos entre agosto y diciembre, la temporada empezó este año
        if current_month >= 8:
            start_year = current_year
            end_year = current_year + 1
        else:
            # Si estamos entre enero y julio, la temporada empezó el año anterior
            start_year = current_year - 1
            end_year = current_year

        # Formatear como los últimos dos dígitos de cada año
        season_code = f"{str(start_year)[-2:]}{str(end_year)[-2:]}"

        logger.info(
            f"Temporada actual detectada: {start_year}-{end_year} (código: {season_code})"
        )
        return season_code

    def download_league_data(self, league_code, season=None, output_dir=None):
        """
        Descarga datos de una liga específica.

        Args:
            league_code (str): Código de la liga (ej: 'SP1', 'E0')
            season (str, optional): Temporada en formato 'YXYY' (ej: '2425').
                                  Si no se especifica, usa la temporada actual.
            output_dir (str, optional): Directorio de salida. Por defecto 'files/datasets/input/'

        Returns:
            str: Ruta del archivo descargado o None si hay error
        """
        if season is None:
            season = self.get_current_season()

        if output_dir is None:
            # Obtener directorio base del proyecto
            base_dir = Path(__file__).parent.parent
            output_dir = base_dir / "files" / "datasets" / "input"
        else:
            output_dir = Path(output_dir)

        # Crear directorio si no existe
        output_dir.mkdir(parents=True, exist_ok=True)

        # Construir URL
        url = f"{self.base_url}/{season}/{league_code}.csv"
        filename = f"{league_code}.csv"
        output_path = output_dir / filename

        logger.info(f"Descargando datos de: {url}")

        try:
            # Realizar petición
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Verificar que el contenido es CSV válido
            if not response.text.strip():
                logger.error("El archivo descargado está vacío")
                return None

            # Verificar que tiene encabezados CSV típicos
            first_line = response.text.split("\n")[0]
            if not any(
                header in first_line.upper()
                for header in ["DATE", "HOMETEAM", "AWAYTEAM"]
            ):
                logger.warning(
                    "El archivo puede no ser un CSV de datos de fútbol válido"
                )

            # Guardar archivo
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write(response.text)

            logger.info(f"Datos descargados exitosamente: {output_path}")
            logger.info(f"Tamaño del archivo: {len(response.text)} caracteres")

            return str(output_path)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error al descargar {url}: {e}")
            return None
        except IOError as e:
            logger.error(f"Error al guardar el archivo {output_path}: {e}")
            return None

    def download_multiple_leagues(self, leagues, season=None, output_dir=None):
        """
        Descarga datos de múltiples ligas.

        Args:
            leagues (list): Lista de nombres de ligas o códigos
            season (str, optional): Temporada en formato 'YXYY'
            output_dir (str, optional): Directorio de salida

        Returns:
            dict: Diccionario con resultados {liga: ruta_archivo o None}
        """
        results = {}

        for league in leagues:
            # Convertir nombre de liga a código si es necesario
            if league.lower() in LEAGUE_CODES:
                league_code = LEAGUE_CODES[league.lower()]
            else:
                league_code = league.upper()

            logger.info(f"Descargando liga: {league} (código: {league_code})")
            result = self.download_league_data(league_code, season, output_dir)
            results[league] = result

        return results


# Inicializar la clase
football_data_downloader = FootballDataDownloader()
