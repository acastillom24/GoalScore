# Football Prediction Toolkit - Documentación Completa

## 📋 Descripción General

Este toolkit modular de predicción de fútbol te permite:

- **Predecir resultados** de partidos específicos
- **Calcular intervalos de goles** con intervalos de confianza  
- **Obtener recomendaciones de apuestas** basadas en modelos estadísticos
- **Analizar ratings de equipos** usando el sistema Elo

## 🏗️ Estructura del Proyecto

```
football_prediction/
├── predictor.py              # Código principal modular
├── example_usage.py          # Script de ejemplo
├── data/
│   ├── spain.csv            # Datos de La Liga
│   ├── matches.csv          # Datos generales
│   └── ejemplo.csv          # Datos de ejemplo
├── README.md               # Esta documentación
└── requirements.txt        # Dependencias
```

## 🔧 Instalación y Configuración

### Dependencias Requeridas

```bash
pip install pandas numpy scipy scikit-learn statsmodels joblib
```

### Formato de Datos Esperado

Tu archivo CSV debe tener estas columnas (formato football-data.co.uk):

| Columna | Descripción | Ejemplo |
|---------|-------------|---------|
| Date | Fecha del partido | 01/08/23 |
| HomeTeam | Equipo local | Real Madrid |
| AwayTeam | Equipo visitante | Barcelona |
| FTHG | Goles equipo local | 2 |
| FTAG | Goles equipo visitante | 1 |

**Columnas alternativas aceptadas:**

- `HG` en lugar de `FTHG`
- `AG` en lugar de `FTAG`

## 🚀 Uso Básico

### Ejemplo Rápido

```python
from predictor import FootballPredictor

# 1. Cargar datos y entrenar modelos
predictor = FootballPredictor('data/spain.csv')
predictor.train_all_models()

# 2. Predecir un partido específico
resultado = predictor.predict_match('Real Madrid', 'Barcelona')

print(f"Goles esperados: {resultado['expected_goals']['total']:.1f}")
print(f"Más de 2.5 goles: {resultado['goal_intervals']['over_2_5']:.1%}")

# 3. Obtener recomendaciones de apuestas
recomendaciones = predictor.get_goal_recommendations('Real Madrid', 'Barcelona')
print(f"Over 2.5: {recomendaciones['betting_tips']['over_2_5']}")
```

### Ejemplo Avanzado con Análisis Detallado

```python
# Predicción detallada
prediction = predictor.predict_match('Real Madrid', 'Barcelona', detailed=True)

# Acceder a diferentes componentes
goles = prediction['expected_goals']
probabilidades = prediction['outcome_probabilities'] 
intervalos = prediction['goal_intervals']
confianza = prediction['confidence_interval']

print(f"Rango probable de goles: {confianza['likely_range']}")
print(f"Rating Elo - Local: {prediction['detailed']['elo_ratings']['home']}")
```

## 📊 Modelos Incluidos

### 1. Sistema de Rating Elo

- **Propósito**: Evalúa la fuerza relativa de los equipos
- **Ventaja local**: 50 puntos por defecto
- **Factor K**: 20 (ajustable)
- **Rating inicial**: 1500

### 2. Modelo Poisson GLM

- **Propósito**: Predice el número de goles esperados
- **Base**: Regresión de Poisson con efectos fijos por equipo
- **Variables**: Ataque del equipo local/visitante, defensa del oponente

### 3. Simulación Monte Carlo

- **Propósito**: Genera distribuciones de probabilidad
- **Simulaciones**: 10,000 por defecto
- **Salida**: Intervalos de confianza y probabilidades de goles

## 🎯 Interpretación de Resultados

### Goles Esperados

```python
'expected_goals': {
    'home': 1.85,      # Goles esperados equipo local
    'away': 1.12,      # Goles esperados equipo visitante  
    'total': 2.97      # Total de goles esperados
}
```

### Probabilidades de Resultado

```python
'outcome_probabilities': {
    'home_win': 0.487,  # 48.7% probabilidad victoria local
    'draw': 0.289,      # 28.9% probabilidad empate
    'away_win': 0.224   # 22.4% probabilidad victoria visitante
}
```

### Intervalos de Goles

```python
'goal_intervals': {
    'under_1_5': 0.186,  # 18.6% menos de 1.5 goles
    'under_2_5': 0.423,  # 42.3% menos de 2.5 goles
    'over_2_5': 0.577,   # 57.7% más de 2.5 goles
    'over_3_5': 0.334    # 33.4% más de 3.5 goles
}
```

### Intervalo de Confianza

```python
'confidence_interval': {
    'total_goals_mean': 2.97,           # Media esperada
    'total_goals_std': 1.72,            # Desviación estándar
    'likely_range': (1.3, 4.7)         # Rango probable (±1 σ)
}
```

## 💡 Recomendaciones de Apuestas

El sistema proporciona tips automáticos basados en umbrales de confianza:

```python
'betting_tips': {
    'over_2_5': 'YES',    # Apostar si probabilidad > 55%
    'under_2_5': 'NO',    # No apostar si probabilidad < 55%
    'over_1_5': 'YES',    # Apostar si probabilidad > 70%
    'btts': 'YES'         # Ambos marcan si cada equipo > 0.8 goles
}
```

## ⚙️ Configuración Avanzada

### Personalizar Parámetros del Modelo Elo

```python
from predictor import EloRatingModel

# Modelo Elo personalizado
elo_custom = EloRatingModel(
    k_factor=30,           # Mayor sensibilidad a resultados recientes
    initial_rating=1600,   # Rating inicial más alto
    home_advantage=75      # Mayor ventaja local
)

predictor.elo_model = elo_custom
```

### Ajustar Simulación Monte Carlo

```python
from predictor import MonteCarloSimulator

# Más simulaciones para mayor precisión
mc_custom = MonteCarloSimulator(n_simulations=50000)
predictor.monte_carlo = mc_custom
```

## 📈 Casos de Uso Específicos

### 1. Análisis de Over/Under

```python
# Para apuestas de goles totales
def analizar_over_under(predictor, home, away):
    pred = predictor.predict_match(home, away)
    
    over_25 = pred['goal_intervals']['over_2_5']
    under_25 = pred['goal_intervals']['under_2_5'] 
    
    if over_25 > 0.6:
        return f"OVER 2.5 ({over_25:.1%})"
    elif under_25 > 0.6:
        return f"UNDER 2.5 ({under_25:.1%})"
    else:
        return "No apostar - muy incierto"
```

### 2. Encontrar Valor en las Cuotas

```python
def encontrar_valor(predictor, home, away, cuota_over25):
    pred = predictor.predict_match(home, away)
    prob_over = pred['goal_intervals']['over_2_5']
    
    probabilidad_implicita = 1 / cuota_over25
    
    if prob_over > probabilidad_implicita * 1.1:  # 10% margen
        return f"VALOR ENCONTRADO: Apostar Over 2.5"
    else:
        return "Sin valor - pasar"
```

### 3. Análisis de Equipos Específicos

```python
def analizar_equipo(predictor, equipo):
    matches = predictor.matches
    
    # Partidos como local
    local = matches[matches['home_team'] == equipo]
    goles_local = local['home_goals'].mean()
    
    # Partidos como visitante  
    visitante = matches[matches['away_team'] == equipo]
    goles_visitante = visitante['away_goals'].mean()
    
    rating = predictor.elo_model.get_rating(equipo)
    
    return {
        'goles_como_local': goles_local,
        'goles_como_visitante': goles_visitante,
        'rating_elo': rating
    }
```

## 🔍 Solución de Problemas

### Error: "Team not found"

- Verifica que los nombres de equipos coincidan exactamente con los del CSV
- Usa `predictor.matches['home_team'].unique()` para ver equipos disponibles

### Error: "Model not fitted"

- Asegúrate de llamar `predictor.train_all_models()` antes de hacer predicciones

### Predicciones poco realistas

- Verifica que tengas suficientes datos (mínimo 50-100 partidos)
- Los equipos nuevos usan promedios de liga hasta que tengan suficiente historial

### Rendimiento lento

- Reduce el número de simulaciones Monte Carlo
- Usa menos datos históricos para entrenamiento más rápido

## 🎛️ Parámetros de Configuración

```python
# config.py - Archivo de configuración
CONFIG = {
    'elo': {
        'k_factor': 20,
        'initial_rating': 1500,
        'home_advantage': 50
    },
    'monte_carlo': {
        'n_simulations': 10000,
        'max_goals': 8
    },
    'betting': {
        'over_under_threshold': 0.55,
        'btts_threshold': 0.8,
        'confidence_threshold': 0.70
    },
    'data': {
        'min_matches_for_prediction': 10,
        'recent_form_games': 5
    }
}
```

## 📚 Referencias y Fuentes de Datos

### Fuentes de Datos Recomendadas

- [Football-Data.co.uk](https://www.football-data.co.uk/) - Datos históricos gratuitos
- FBref.com - Estadísticas avanzadas
- ESPN Soccer - Resultados en tiempo real

### Literatura Relevante

- Dixon, M. & Coles, S. (1997) - Modelling Association Football Scores
- Maher, M. (1982) - Modelling Association Football Scores
- Karlis, D. & Ntzoufras, I. (2003) - Analysis of Sports Data by Using Bivariate Poisson Models

## 🤝 Contribuir

Para mejorar el toolkit:

1. Fork del repositorio
2. Crear branch para tu feature
3. Añadir tests si es posible
4. Submit pull request

## 📄 Licencia

MIT License - usar libremente para proyectos personales y comerciales.

---

**¿Preguntas?** Abre un issue en GitHub o consulta la documentación adicional en los comentarios del código.
