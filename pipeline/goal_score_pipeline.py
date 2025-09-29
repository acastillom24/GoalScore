"""
Ejemplo de uso del Football Prediction Toolkit
==============================================

Este script demuestra cómo usar el toolkit modular para predecir
resultados de partidos de fútbol y intervalos de goles.
"""

import pandas as pd

# Importar el predictor (asumiendo que está en el mismo directorio)
from predictor import FootballPredictor


def main():
    """Función principal con ejemplos de uso."""

    print("🏈 Football Prediction Toolkit")
    print("=" * 50)

    # 1. Inicializar el predictor
    print("\n📊 Cargando datos...")

    # Reemplaza 'data/spain.csv' con la ruta a tu archivo CSV
    # El archivo debe tener las columnas: Date, HomeTeam, AwayTeam, FTHG, FTAG
    data_path = "data/spain.csv"  # ⚠️ Cambia esta ruta

    try:
        predictor = FootballPredictor(data_path)
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {data_path}")
        print("Por favor verifica la ruta del archivo CSV")
        return
    except Exception as e:
        print(f"❌ Error cargando datos: {str(e)}")
        return

    # 2. Entrenar todos los modelos
    print("\n🤖 Entrenando modelos...")
    predictor.train_all_models()

    # 3. Hacer predicciones para equipos específicos
    equipos_ejemplo = [
        ("Real Madrid", "Barcelona"),
        ("Atletico Madrid", "Sevilla"),
        ("Valencia", "Villarreal"),
    ]

    for home_team, away_team in equipos_ejemplo:
        print(f"\n" + "=" * 60)
        print(f"🏆 PREDICCIÓN: {home_team} vs {away_team}")
        print(f"=" * 60)

        try:
            # Obtener predicción detallada
            prediction = predictor.predict_match(home_team, away_team, detailed=True)

            # Mostrar resultados principales
            mostrar_prediccion_detallada(prediction)

            # Obtener recomendaciones de apuestas
            recommendations = predictor.get_goal_recommendations(home_team, away_team)
            mostrar_recomendaciones(recommendations)

        except Exception as e:
            print(f"❌ Error en predicción: {str(e)}")
            print("Verifica que los nombres de los equipos sean correctos")

    # 4. Análisis interactivo
    print(f"\n" + "=" * 60)
    print("🎯 PREDICCIÓN PERSONALIZADA")
    print(f"=" * 60)

    # Ejemplo de predicción personalizada (descomenta para usar)
    """
    while True:
        try:
            home = input("\nEquipo local (o 'quit' para salir): ").strip()
            if home.lower() == 'quit':
                break
                
            away = input("Equipo visitante: ").strip()
            
            prediction = predictor.predict_match(home, away)
            mostrar_prediccion_simple(prediction)
            
            recommendations = predictor.get_goal_recommendations(home, away)
            mostrar_recomendaciones(recommendations)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    """


def mostrar_prediccion_detallada(prediction):
    """Mostrar predicción detallada de un partido."""

    # Goles esperados
    print("⚽ Goles Esperados:")
    goals = prediction["expected_goals"]
    print(f"  🏠 Local: {goals['home']:.2f}")
    print(f"  ✈️  Visitante: {goals['away']:.2f}")
    print(f"  📊 Total: {goals['total']:.2f}")

    # Probabilidades de resultado
    print("\n🎲 Probabilidades de Resultado:")
    probs = prediction["outcome_probabilities"]
    print(f"  🏠 Victoria Local: {probs['home_win']:.1%}")
    print(f"  🤝 Empate: {probs['draw']:.1%}")
    print(f"  ✈️  Victoria Visitante: {probs['away_win']:.1%}")

    # Intervalos de goles
    print("\n📈 Intervalos de Goles:")
    intervals = prediction["goal_intervals"]
    print(f"  Más de 1.5 goles: {intervals['over_1_5']:.1%}")
    print(f"  Más de 2.5 goles: {intervals['over_2_5']:.1%}")
    print(f"  Más de 3.5 goles: {intervals['over_3_5']:.1%}")
    print(f"  Menos de 2.5 goles: {intervals['under_2_5']:.1%}")

    # Intervalo de confianza
    print("\n📊 Intervalo de Confianza:")
    ci = prediction["confidence_interval"]
    print(f"  Rango probable: {ci['likely_range'][0]} - {ci['likely_range'][1]} goles")
    print(
        f"  Media ± Desviación: {ci['total_goals_mean']:.1f} ± {ci['total_goals_std']:.1f}"
    )

    # Información adicional si está disponible
    if "detailed" in prediction:
        print("\n🔍 Detalles Adicionales:")
        detailed = prediction["detailed"]

        if "elo_ratings" in detailed:
            elo = detailed["elo_ratings"]
            print(
                f"  Ratings Elo - Local: {elo['home']:.0f}, Visitante: {elo['away']:.0f}"
            )
            print(f"  Diferencia de Rating: {elo['difference']:.0f}")

        if "sample_scores" in detailed:
            scores = detailed["sample_scores"]
            print(f"  Resultados simulados (muestra): {scores[:5]}")


def mostrar_prediccion_simple(prediction):
    """Mostrar predicción simplificada."""
    goals = prediction["expected_goals"]
    probs = prediction["outcome_probabilities"]
    intervals = prediction["goal_intervals"]

    print(
        f"\n⚽ Goles esperados: {goals['home']:.1f} - {goals['away']:.1f} (Total: {goals['total']:.1f})"
    )
    print(f"🎲 Resultado más probable: ", end="")

    max_prob = max(probs["home_win"], probs["draw"], probs["away_win"])
    if max_prob == probs["home_win"]:
        print(f"Victoria Local ({probs['home_win']:.1%})")
    elif max_prob == probs["draw"]:
        print(f"Empate ({probs['draw']:.1%})")
    else:
        print(f"Victoria Visitante ({probs['away_win']:.1%})")

    print(f"📈 Más de 2.5 goles: {intervals['over_2_5']:.1%}")


def mostrar_recomendaciones(recommendations):
    """Mostrar recomendaciones de apuestas."""
    print("\n💡 Recomendaciones de Apuestas:")

    # Goles más probables
    print(f"  🎯 Goles más probables: {recommendations['most_likely_total_goals']}")
    print(
        f"  📊 Rango de confianza: {recommendations['confidence_range'][0]}-{recommendations['confidence_range'][1]} goles"
    )

    # Tips de apuestas
    print("\n💰 Tips de Apuestas:")
    tips = recommendations["betting_tips"]
    probs = recommendations["probabilities"]

    for bet_type, tip in tips.items():
        prob_str = probs.get(bet_type, "N/A")
        emoji = "✅" if tip == "YES" else "❌"

        bet_name = {
            "over_2_5": "Más de 2.5 goles",
            "under_2_5": "Menos de 2.5 goles",
            "over_1_5": "Más de 1.5 goles",
            "btts": "Ambos equipos marcan",
        }.get(bet_type, bet_type)

        print(f"  {emoji} {bet_name}: {tip} ({prob_str})")


def analizar_temporada(predictor):
    """Análisis adicional de la temporada."""
    print("\n📊 ANÁLISIS DE TEMPORADA")
    print("=" * 40)

    if predictor.matches is not None:
        matches = predictor.matches

        # Estadísticas generales
        total_matches = len(matches)
        avg_goals = (matches["home_goals"] + matches["away_goals"]).mean()

        print(f"Total de partidos: {total_matches}")
        print(f"Promedio de goles por partido: {avg_goals:.2f}")

        # Distribución de resultados
        home_wins = len(matches[matches["home_goals"] > matches["away_goals"]])
        draws = len(matches[matches["home_goals"] == matches["away_goals"]])
        away_wins = len(matches[matches["home_goals"] < matches["away_goals"]])

        print(f"\nDistribución de resultados:")
        print(f"  Victorias locales: {home_wins} ({home_wins/total_matches:.1%})")
        print(f"  Empates: {draws} ({draws/total_matches:.1%})")
        print(f"  Victorias visitantes: {away_wins} ({away_wins/total_matches:.1%})")

        # Top equipos por rating Elo
        print(f"\n🏆 Top 10 Equipos (Rating Elo):")
        top_teams = sorted(
            predictor.elo_model.ratings.items(), key=lambda x: x[1], reverse=True
        )[:10]

        for i, (team, rating) in enumerate(top_teams, 1):
            print(f"  {i:2d}. {team}: {rating:.0f}")


def validacion_datos(data_path):
    """Validar que el archivo de datos tenga el formato correcto."""
    try:
        df = pd.read_csv(data_path)

        # Verificar columnas necesarias
        required_cols = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        alt_cols = {"Date", "HomeTeam", "AwayTeam", "HG", "AG"}  # Nombres alternativos

        available_cols = set(df.columns)

        if required_cols.issubset(available_cols):
            print("✅ Formato de datos correcto (football-data.co.uk)")
            return True
        elif alt_cols.issubset(available_cols):
            print("✅ Formato de datos correcto (alternativo)")
            return True
        else:
            print("❌ Formato de datos incorrecto")
            print(f"Columnas encontradas: {list(df.columns)}")
            print(f"Columnas necesarias: {list(required_cols)} o {list(alt_cols)}")
            return False

    except Exception as e:
        print(f"❌ Error validando datos: {str(e)}")
        return False


def crear_datos_ejemplo():
    """Crear un archivo de datos de ejemplo si no existe."""
    import os

    if not os.path.exists("data"):
        os.makedirs("data")

    # Datos de ejemplo (temporada ficticia)
    datos_ejemplo = {
        "Date": ["01/08/23", "02/08/23", "03/08/23", "04/08/23", "05/08/23"] * 20,
        "HomeTeam": [
            "Real Madrid",
            "Barcelona",
            "Atletico Madrid",
            "Sevilla",
            "Valencia",
        ]
        * 20,
        "AwayTeam": [
            "Barcelona",
            "Atletico Madrid",
            "Sevilla",
            "Valencia",
            "Real Madrid",
        ]
        * 20,
        "FTHG": [2, 1, 3, 0, 2] * 20,
        "FTAG": [1, 1, 1, 2, 0] * 20,
    }

    df_ejemplo = pd.DataFrame(datos_ejemplo)
    df_ejemplo.to_csv("data/ejemplo.csv", index=False)

    print("✅ Archivo de ejemplo creado en 'data/ejemplo.csv'")
    return "data/ejemplo.csv"


# Función para ejecutar diferentes escenarios
def ejecutar_escenarios():
    """Ejecutar diferentes escenarios de predicción."""

    print("🚀 INICIANDO ANÁLISIS DE FÚTBOL")
    print("=" * 50)

    # Intentar cargar datos reales, si no crear ejemplo
    data_paths = [
        "data/spain.csv",
        "data/laliga.csv",
        "data/matches.csv",
        "data/football_data.csv",
    ]

    data_path = None
    for path in data_paths:
        if validacion_datos(path):
            data_path = path
            break

    if not data_path:
        print("📝 No se encontraron datos válidos, creando archivo de ejemplo...")
        data_path = crear_datos_ejemplo()

    # Ejecutar análisis principal
    main_with_path(data_path)


def main_with_path(data_path):
    """Función principal con ruta específica."""
    try:
        # Inicializar predictor
        predictor = FootballPredictor(data_path)

        # Entrenar modelos
        print("\n🤖 Entrenando modelos de predicción...")
        predictor.train_all_models()

        # Análisis de temporada
        analizar_temporada(predictor)

        # Obtener equipos disponibles
        equipos = sorted(predictor.matches["home_team"].unique())
        print(f"\n📋 Equipos disponibles ({len(equipos)}):")
        for i, equipo in enumerate(equipos[:15], 1):  # Mostrar solo los primeros 15
            print(f"  {i:2d}. {equipo}")
        if len(equipos) > 15:
            print(f"  ... y {len(equipos) - 15} equipos más")

        # Predicciones de ejemplo
        if len(equipos) >= 4:
            equipos_ejemplo = [
                (equipos[0], equipos[1]),
                (
                    (equipos[2], equipos[3])
                    if len(equipos) > 3
                    else (equipos[0], equipos[2])
                ),
            ]

            for home_team, away_team in equipos_ejemplo:
                print(f"\n" + "=" * 60)
                print(f"🏆 PREDICCIÓN: {home_team} vs {away_team}")
                print(f"=" * 60)

                try:
                    prediction = predictor.predict_match(
                        home_team, away_team, detailed=True
                    )
                    mostrar_prediccion_detallada(prediction)

                    recommendations = predictor.get_goal_recommendations(
                        home_team, away_team
                    )
                    mostrar_recomendaciones(recommendations)

                except Exception as e:
                    print(f"❌ Error en predicción: {str(e)}")

        print(f"\n✅ Análisis completado exitosamente!")

    except Exception as e:
        print(f"❌ Error general: {str(e)}")


if __name__ == "__main__":
    # Ejecutar escenarios automáticamente
    ejecutar_escenarios()

    # Ejemplo de uso programático
    """
    # Para usar en tus propios scripts:
    
    from predictor import FootballPredictor
    
    # Cargar y entrenar
    predictor = FootballPredictor('tu_archivo.csv')
    predictor.train_all_models()
    
    # Predecir partido específico
    result = predictor.predict_match('Real Madrid', 'Barcelona')
    print(f"Goles esperados: {result['expected_goals']['total']:.1f}")
    print(f"Más de 2.5 goles: {result['goal_intervals']['over_2_5']:.1%}")
    
    # Obtener recomendaciones
    tips = predictor.get_goal_recommendations('Real Madrid', 'Barcelona')
    print(f"Recomendación Over 2.5: {tips['betting_tips']['over_2_5']}")
    """
