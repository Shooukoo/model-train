import os
import requests
from duckduckgo_search import DDGS
from time import sleep

def descargar_imagenes_ddg():
    # Carpeta destino (usamos la misma estructura temporal)
    output_dir = os.path.join('datasets', 'frutas', 'temp_raw')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- TÉRMINOS MEJORADOS (Clave para evitar fotos basura) ---
    # Usamos términos muy específicos en inglés para evitar recetas de cocina o publicidad.
    queries = {
        "fresh ripe blackberry fruit on bush Rubus fruticosus"
    }

    # Cuántas fotos intentar bajar por término (DuckDuckGo da unas 20-30 buenas)
    max_results_per_query = 60 
    
    print(f"--- INICIANDO DESCARGA MEJORADA EN: {output_dir} ---\n")
    
    total_descargadas = 0

    # Inicializamos el motor de búsqueda
    with DDGS() as ddgs:
        for  query in queries:
            print(f"Busca: '{query}' ")
            
            # Buscamos los links de las imágenes
            results = ddgs.images(
                keywords=query,
                region="wt-wt", # Región mundial
                safesearch="off",
                max_results=max_results_per_query
            )

            count_local = 0
            for r in results:
                image_url = r['image']
                
                try:
                    # Hacemos la petición de descarga con un timeout para no trabarnos
                    response = requests.get(image_url, timeout=5)
                    
                    # Verificamos que sea una respuesta válida (código 200)
                    if response.status_code == 200:
                        # Guardamos el archivo
                        filename = f"zarzamoras_{count_local}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        # Verificación simple: Si el archivo pesa menos de 5KB, probablemente sea un icono basura
                        if os.path.getsize(filepath) < 5120: 
                            os.remove(filepath)
                        else:
                            count_local += 1
                            total_descargadas += 1
                            print(f"  -> [OK] {filename}")

                except Exception:
                    # Si falla un link específico, simplemente pasamos al siguiente
                    pass

                # Pequeña pausa para no saturar
                if count_local >= 60: # Límite de seguridad por categoría
                    break
            
            print(f"  Terminado 'zarzamoras': {count_local} imágenes válidas.\n")
            sleep(1) # Descanso entre categorías

    print("="*40)
    print(f"DESCARGA FINALIZADA. Total aproximado: {total_descargadas} imágenes.")
    print(f"Revisa la carpeta: {output_dir}")
    print("="*40)

if __name__ == "__main__":
    descargar_imagenes_ddg()