"""
Script para obtener los próximos partidos de La Liga desde Google Search
Extrae la información del widget de deportes de Google
"""

import time
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def configurar_driver():
    """Configura el driver de Chrome"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--lang=es')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def obtener_proximos_partidos_laliga():
    """
    Obtiene los próximos partidos de La Liga desde Google Search
    """
    print("⚽ PRÓXIMOS PARTIDOS DE LA LIGA EA SPORTS")
    print("=" * 90)
    print(f"🕐 Consulta realizada: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    
    url = "https://www.google.com/search?q=la+liga+partidos"
    
    driver = None
    
    try:
        print("🔍 Iniciando consulta a Google...\n")
        driver = configurar_driver()
        
        print("🌐 Cargando resultados de búsqueda...\n")
        driver.get(url)
        
        # Esperar a que cargue el contenido
        time.sleep(3)
        
        # Intentar aceptar cookies si aparece el diálogo
        try:
            accept_button = driver.find_element(By.XPATH, "//button[contains(., 'Aceptar') or contains(., 'Accept')]")
            accept_button.click()
            time.sleep(1)
        except:
            pass
        
        wait = WebDriverWait(driver, 10)
        
        # Buscar el widget de partidos de Google
        # Google usa diferentes estructuras, intentaremos varios selectores
        
        partidos_encontrados = []
        
        # Método 1: Buscar por div de eventos deportivos
        try:
            # Buscar contenedor de partidos
            contenedor = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-entityname='LaLiga'], div[class*='sports'], div[class*='match']"))
            )
            
            # Buscar todos los partidos
            partidos = driver.find_elements(By.CSS_SELECTOR, "div[class*='match'], div[jsname], tr[class*='match'], div[data-match]")
            
            for partido in partidos[:15]:
                try:
                    # Intentar obtener equipos
                    equipos = partido.find_elements(By.CSS_SELECTOR, "div[class*='team'], span[class*='team'], div[class*='participant']")
                    
                    if len(equipos) >= 2:
                        local = equipos[0].text.strip()
                        visitante = equipos[1].text.strip()
                        
                        # Buscar fecha/hora
                        fecha_hora = ""
                        try:
                            tiempo = partido.find_element(By.CSS_SELECTOR, "div[class*='time'], span[class*='time'], div[class*='date']")
                            fecha_hora = tiempo.text.strip()
                        except:
                            try:
                                tiempo = partido.find_element(By.TAG_NAME, "time")
                                fecha_hora = tiempo.text.strip()
                            except:
                                pass
                        
                        if local and visitante:
                            partidos_encontrados.append({
                                'local': local,
                                'visitante': visitante,
                                'fecha_hora': fecha_hora
                            })
                
                except Exception as e:
                    continue
        
        except Exception as e:
            print(f"⚠️ Método 1 falló: {str(e)}")
        
        # Método 2: Buscar tablas de resultados
        if not partidos_encontrados:
            try:
                tablas = driver.find_elements(By.TAG_NAME, "table")
                
                for tabla in tablas:
                    filas = tabla.find_elements(By.TAG_NAME, "tr")
                    
                    for fila in filas:
                        celdas = fila.find_elements(By.TAG_NAME, "td")
                        
                        if len(celdas) >= 3:
                            texto = fila.text
                            if "vs" in texto.lower() or "-" in texto:
                                partidos_encontrados.append({
                                    'texto_completo': texto
                                })
            except Exception as e:
                print(f"⚠️ Método 2 falló: {str(e)}")
        
        # Método 3: Extraer todo el texto y buscar patrones
        if not partidos_encontrados:
            try:
                texto_completo = driver.find_element(By.TAG_NAME, "body").text
                lineas = texto_completo.split('\n')
                
                for i, linea in enumerate(lineas):
                    if 'vs' in linea.lower() or ' - ' in linea:
                        # Buscar contexto (fecha/hora cercana)
                        contexto = ""
                        if i > 0:
                            contexto = lineas[i-1]
                        
                        partidos_encontrados.append({
                            'linea': linea.strip(),
                            'contexto': contexto
                        })
            except Exception as e:
                print(f"⚠️ Método 3 falló: {str(e)}")
        
        # Mostrar resultados
        if partidos_encontrados:
            print(f"✅ Se encontraron {len(partidos_encontrados)} partidos\n")
            print("=" * 90)
            print("PRÓXIMOS PARTIDOS")
            print("=" * 90 + "\n")
            
            for i, partido in enumerate(partidos_encontrados, 1):
                if 'local' in partido:
                    print(f"{i}. ⚽ {partido['local']:30} vs {partido['visitante']:30}")
                    if partido['fecha_hora']:
                        print(f"   🕐 {partido['fecha_hora']}")
                elif 'texto_completo' in partido:
                    print(f"{i}. {partido['texto_completo']}")
                elif 'linea' in partido:
                    if partido['contexto']:
                        print(f"{i}. 📅 {partido['contexto']}")
                    print(f"   ⚽ {partido['linea']}")
                print()
            
            print("=" * 90)
        else:
            print("⚠️ No se pudieron extraer los partidos del widget de Google.\n")
            print("Esto puede deberse a:")
            print("  • Google cambió la estructura de su página")
            print("  • Se requiere interacción adicional")
            print("  • El contenido se carga de forma diferente\n")
            mostrar_alternativas()
            
    except Exception as e:
        print(f"❌ Error: {str(e)}\n")
        mostrar_alternativas()
        
    finally:
        if driver:
            driver.quit()
            print("\n🔒 Navegador cerrado.")

def mostrar_alternativas():
    """Muestra opciones alternativas"""
    print("📌 ALTERNATIVAS RECOMENDADAS:")
    print("   • https://www.google.com/search?q=la+liga")
    print("   • https://www.laliga.com/laliga-easports/calendario")
    print("   • https://www.espn.com.co/futbol/calendario/_/liga/esp.1")
    print("   • https://www.flashscore.es/futbol/espana/laliga-ea-sports/")

if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("  REQUISITOS:")
    print("  - pip install selenium")
    print("  - ChromeDriver instalado y en PATH")
    print("=" * 90 + "\n")
    
    try:
        obtener_proximos_partidos_laliga()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n💡 SOLUCIONES:")
        print("   1. Instala Selenium: pip install selenium")
        print("   2. Instala ChromeDriver o usa: pip install webdriver-manager")
        print("\n   Si usas webdriver-manager, agrega al inicio del script:")
        print("   from webdriver_manager.chrome import ChromeDriverManager")
        print("   driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)")
        mostrar_alternativas()
    
    print("\n💡 NOTA:")
    print("   Google actualiza constantemente su estructura HTML.")
    print("   Si el script no funciona, la forma más confiable es consultar")
    print("   directamente en: https://www.google.com/search?q=la+liga")
