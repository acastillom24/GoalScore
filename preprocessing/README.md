# 📊 Preprocessing - GoalScore

Esta carpeta contiene los módulos de preprocesamiento de datos para el proyecto GoalScore. Su función principal es descargar, limpiar y preparar los datos de fútbol necesarios para el análisis y modelado predictivo.

## 📁 Estructura de Archivos

```text
preprocessing/
├── a_get_data.py    # Descargador de datos de football-data.co.uk
└── README.md        # Documentación del módulo
```

## 🔧 Módulos Disponibles

### `a_get_data.py` - Descargador de Datos de Fútbol

**Descripción:** Módulo especializado para descargar datos históricos de partidos de fútbol desde [football-data.co.uk](https://www.football-data.co.uk/). Incluye soporte para las principales ligas europeas y detección automática de temporadas.

#### 🌟 Características Principales

- **Descarga automática** de datos de las principales ligas europeas
- **Detección inteligente de temporadas** basada en la fecha actual
- **Soporte para múltiples ligas** simultáneamente
- **Configuración flexible** de directorios de salida
- **Manejo robusto de errores** y logging detallado
- **Interface de línea de comandos** con argumentos configurables

#### 🏆 Ligas Soportadas

| Liga | Código | País |
|------|--------|------|
| La Liga | SP1 | 🇪🇸 España |
| Premier League | E0 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra |
| Bundesliga | D1 | 🇩🇪 Alemania |
| Serie A | I1 | 🇮🇹 Italia |
| Ligue 1 | F1 | 🇫🇷 Francia |
| Eredivisie | N1 | 🇳🇱 Países Bajos |
| Primeira Liga | P1 | 🇵🇹 Portugal |
| Pro League | B1 | 🇧🇪 Bélgica |
| Süper Lig | T1 | 🇹🇷 Turquía |
| Super League | G1 | 🇬🇷 Grecia |

#### 📋 Argumentos de Línea de Comandos

| Argumento | Descripción | Ejemplo |
|-----------|-------------|---------|
| `--league, -l` | Liga(s) a descargar | `--league spain` |
| `--season, -s` | Temporada específica (YXYY) | `--season 2425` |
| `--output-dir, -o` | Directorio de salida | `--output-dir /custom/path` |
| `--list-leagues` | Mostrar ligas disponibles | `--list-leagues` |
| `--verbose, -v` | Información detallada | `--verbose` |
| `--help, -h` | Mostrar ayuda | `--help` |

#### 💻 Ejemplos de Uso

```bash
# Descargar La Liga temporada actual
python preprocessing/a_get_data.py --league spain

# Descargar Premier League temporada específica
python preprocessing/a_get_data.py --league england --season 2324

# Descargar múltiples ligas
python preprocessing/a_get_data.py --league spain england germany

# Ver todas las ligas disponibles
python preprocessing/a_get_data.py --list-leagues

# Descarga con información detallada
python preprocessing/a_get_data.py --league spain --verbose

# Especificar directorio de salida personalizado
python preprocessing/a_get_data.py --league italy --output-dir ./custom_data/
```

#### 🔄 Uso Programático

También puedes usar el módulo directamente en tu código Python:

```python
from preprocessing.a_get_data import FootballDataDownloader

# Crear descargador
downloader = FootballDataDownloader()

# Descargar una liga específica
file_path = downloader.download_league_data('SP1', season='2425')

# Descargar múltiples ligas
results = downloader.download_multiple_leagues(['spain', 'england'])
```

#### 📂 Estructura de Salida

Por defecto, los archivos se guardan en:

```text
files/
└── datasets/
    └── input/
        ├── SP1.csv    # La Liga
        ├── E0.csv     # Premier League
        └── ...
```

#### 🛡️ Manejo de Errores

El módulo incluye manejo robusto de errores para:

- **Errores de red:** Timeouts, conexiones fallidas
- **Archivos no encontrados:** URLs inexistentes o temporadas no disponibles
- **Formato de datos:** Validación de estructura CSV
- **Permisos de archivo:** Problemas de escritura en disco
- **Interrupciones:** Manejo de Ctrl+C durante descargas

#### 📊 Formato de Datos

Los archivos CSV descargados contienen las siguientes columnas típicas:

- `Date`: Fecha del partido
- `HomeTeam`: Equipo local
- `AwayTeam`: Equipo visitante
- `FTHG`: Goles equipo local (tiempo completo)
- `FTAG`: Goles equipo visitante (tiempo completo)
- `FTR`: Resultado final (H/D/A)
- Y muchas más estadísticas detalladas...

## 🚀 Instalación y Configuración

### Dependencias Requeridas

```bash
pip install requests
```

### Configuración del Entorno

El módulo está configurado para trabajar desde el directorio raíz del proyecto:

```bash
cd GoalScore/
python preprocessing/a_get_data.py --help
```

## 📈 Integración con el Pipeline

Este módulo forma parte del pipeline de datos de GoalScore:

1. **Descarga** (a_get_data.py) → Obtiene datos raw de football-data.co.uk
2. **Limpieza** → [Próximos módulos]
3. **Transformación** → [Próximos módulos]
4. **Validación** → [Próximos módulos]

## 🔧 Desarrollo y Contribución

### Estructura de la Clase Principal

```python
class FootballDataDownloader:
    def __init__(self, base_url="https://www.football-data.co.uk/mmz4281")
    def get_current_season(self) -> str
    def download_league_data(self, league_code, season=None, output_dir=None) -> str
    def download_multiple_leagues(self, leagues, season=None, output_dir=None) -> dict
```

### Logging

El módulo usa logging estándar de Python:

- **INFO:** Información general de descarga
- **ERROR:** Errores durante la descarga
- **DEBUG:** Información detallada (con --verbose)

## 📝 Notas Técnicas

- **Detección de temporadas:** La lógica considera que la temporada de fútbol va de agosto a mayo
- **User-Agent:** Configurado para evitar bloqueos por parte del servidor
- **Timeout:** 30 segundos por descarga para evitar cuelgues
- **Encoding:** UTF-8 para soporte de caracteres especiales
- **Session persistente:** Reutiliza conexiones para mejor rendimiento

## 🤝 Próximas Funcionalidades

- [ ] Soporte para más ligas (Championship, Liga 2, etc.)
- [ ] Cache inteligente para evitar descargas duplicadas
- [ ] Validación avanzada de integridad de datos
- [ ] Soporte para descargas incrementales
- [ ] Integración con bases de datos

---

**Autor:** Proyecto GoalScore  
**Última actualización:** Septiembre 2025
