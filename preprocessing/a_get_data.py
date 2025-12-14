#!/usr/bin/env python3
"""
Módulo para descargar datos de fútbol de football-data.co.uk
Descarga archivos CSV de las principales ligas europeas.
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Configurar path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

from functions.football_data_downloader import football_data_downloader
from functions.utils import load_config

config = load_config()

# Mapeo de ligas principales
LEAGUE_CODES = config["configuration"]["league_codes"]


# Crear un función para declarar los argumentos
def parse_args():
    parser = argparse.ArgumentParser(
        description="Descargar datos de fútbol de football-data.co.uk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
            Ejemplos de uso:
                python preprocessing/%(prog)s --league spain                    # Descargar La Liga temporada actual
                python preprocessing/%(prog)s --league england --season 2324    # Descargar Premier League 2023-24
                python preprocessing/%(prog)s --league spain england germany    # Descargar múltiples ligas
                python preprocessing/%(prog)s --list-leagues                    # Mostrar ligas disponibles

            Ligas disponibles:
                {
                    chr(10).join(
                            [f"{name}: {code}" if i == 0 else f"                {name}: {code}" 
                            for i, (name, code) in enumerate(LEAGUE_CODES.items())]
                        )
                }
        """,
    )

    parser.add_argument(
        "--league",
        "-l",
        nargs="+",
        required=True,
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

    return parser.parse_args()


# ====================================================================================
# Ejecución principal
# ====================================================================================
args = parse_args()

# Validación de argumentos
if not args.league:
    logger.error("Debe especificar al menos una liga con --league")
    sys.exit(1)

# Manejar listado de ligas
if args.list_leagues:
    logger.info("Ligas disponibles:")
    for name, code in LEAGUE_CODES.items():
        logger.info(f"  {name}: {code}")
    sys.exit(0)

# Crear descargador
downloader = football_data_downloader

try:
    # Descargar datos
    results = downloader.download_multiple_leagues(
        leagues=args.league, season=args.season, output_dir=args.output_dir
    )

    # Mostrar resultados
    logger.info("\nResultados de descarga:")
    for league, path in results.items():
        if path:
            logger.info(f"✓ {league}: {path}")
        else:
            logger.info(f"✗ {league}: Error en descarga")

    # Estadísticas finales
    successful = sum(1 for path in results.values() if path is not None)
    total = len(results)
    logger.info(f"\nDescargas exitosas: {successful}/{total}")

except KeyboardInterrupt:
    logger.info("Descarga cancelada por el usuario")
except Exception as e:
    logger.error(f"Error inesperado: {e}")
    sys.exit(1)

sys.exit(0)
