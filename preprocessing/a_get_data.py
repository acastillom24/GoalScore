#!/usr/bin/env python3
"""
Módulo para descargar datos de fútbol de football-data.co.uk
Descarga archivos CSV de las principales ligas europeas.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

# Configurar path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Mapeo de ligas principales
LEAGUE_CODES = {
    "spain": "SP1",  # La Liga (Primera División)
    "england": "E0",  # Premier League
    "germany": "D1",  # Bundesliga
    "italy": "I1",  # Serie A
    "france": "F1",  # Ligue 1
    "netherlands": "N1",  # Eredivisie
    "portugal": "P1",  # Primeira Liga
    "belgium": "B1",  # Pro League
    "turkey": "T1",  # Süper Lig
    "greece": "G1",  # Super League
}


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


def main():
    """
    Función principal para ejecutar desde línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Descargar datos de fútbol de football-data.co.uk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
            Ejemplos de uso:
                %(prog)s --league spain                    # Descargar La Liga temporada actual
                %(prog)s --league england --season 2324    # Descargar Premier League 2023-24
                %(prog)s --league spain england germany    # Descargar múltiples ligas
                %(prog)s --list-leagues                    # Mostrar ligas disponibles

            Ligas disponibles:
                {chr(10).join([f"  {name}: {code}" for name, code in LEAGUE_CODES.items()])}
        """,
    )

    parser.add_argument(
        "--league",
        "-l",
        nargs="+",
        help="Liga(s) a descargar. Puede ser nombre (spain, england) o código (SP1, E0)",
    )

    parser.add_argument(
        "--season",
        "-s",
        help="Temporada en formato YXYY (ej: 2425 para 2024-25). Por defecto: temporada actual",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        help="Directorio de salida. Por defecto: files/datasets/input/",
    )

    parser.add_argument(
        "--list-leagues", action="store_true", help="Mostrar ligas disponibles y salir"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Mostrar información detallada"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_leagues:
        print("Ligas disponibles:")
        for name, code in LEAGUE_CODES.items():
            print(f"  {name}: {code}")
        return

    if not args.league:
        parser.error("Debe especificar al menos una liga con --league")

    # Crear descargador
    downloader = FootballDataDownloader()

    try:
        # Descargar datos
        results = downloader.download_multiple_leagues(
            leagues=args.league, season=args.season, output_dir=args.output_dir
        )

        # Mostrar resultados
        print("\nResultados de descarga:")
        for league, path in results.items():
            if path:
                print(f"✓ {league}: {path}")
            else:
                print(f"✗ {league}: Error en descarga")

        # Estadísticas finales
        successful = sum(1 for path in results.values() if path is not None)
        total = len(results)
        print(f"\nDescargas exitosas: {successful}/{total}")

    except KeyboardInterrupt:
        logger.info("Descarga cancelada por el usuario")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
