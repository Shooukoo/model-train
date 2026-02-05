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
    # --- DICCIONARIO DE BÚSQUEDA MASIVA ---
    queries = {
        # --- ROYA (RUST) ---
        "roya_1": "blackberry leaf rust symptoms",
        "roya_2": "Phragmidium violaceum blackberry",  # Nombre científico exacto
        "roya_3": "orange rust gymnoconia nitens blackberry",
        "roya_4": "Rubus fruticosus rust disease",

        # --- BOTRYTIS (MOHO GRIS) ---
        "botrytis_1": "botrytis cinerea blackberry fruit",
        "botrytis_2": "gray mold on blackberry",
        "botrytis_3": "rotten blackberry fruit macro",
        "botrytis_4": "fungal infection rubus fruit",

        # --- MILDEO (DOWNY MILDEW) ---
        "mildeo_1": "Peronospora sparsa blackberry",     # Nombre científico
        "mildeo_2": "downy mildew blackberry leaf symptoms",
        "mildeo_3": "dryberry disease blackberry",       # Otro nombre común
        "mildeo_4": "purple blotches blackberry leaf",   # Síntoma visual

        # --- ANTRACNOSIS ---
        "antracnosis_1": "anthracnose blackberry cane",
        "antracnosis_2": "Elsinoe veneta blackberry",    # Nombre científico
        "antracnosis_3": "bird eye spot blackberry",     # Nombre del síntoma
        "antracnosis_4": "blackberry stem disease",

        # Es vital que la IA distinga esto de una infección
        "dano_sol_1": "white drupelet disorder blackberry",
        "dano_sol_2": "sunscald blackberry fruit",
        "dano_sol_3": "UV damage blackberry fruit",
        "dano_rojo_4": "red cell regression blackberry", # Reversión roja (post-cosecha)

        # La IA necesita saber cómo se ve una planta BIEN para comparar
        "sano_1": "healthy blackberry fruit on plant",
        "sano_2": "Rubus fruticosus healthy leaf",
        "sano_3": "fresh blackberry harvest field",
        "sano_4": "green blackberry leaves",
        
        # --- GENERAL/OTRAS (Para variedad) ---
        "fusarium": "fusarium wilt blackberry",
        "verticillium": "verticillium wilt blackberry",
        "virus": "blackberry yellow vein disease"
    }

    # Cuántas fotos intentar bajar por término (DuckDuckGo da unas 20-30 buenas)
    max_results_per_query = 60 
    
    print(f"--- INICIANDO DESCARGA MEJORADA EN: {output_dir} ---\n")
    
    total_descargadas = 0

    # Inicializamos el motor de búsqueda
    with DDGS() as ddgs:
        for categoria, query in queries.items():
            print(f"Busca: '{query}' ({categoria})...")
            
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
                        filename = f"{categoria}_{count_local}.jpg"
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
            
            print(f"  Terminado '{categoria}': {count_local} imágenes válidas.\n")
            sleep(1) # Descanso entre categorías

    print("="*40)
    print(f"DESCARGA FINALIZADA. Total aproximado: {total_descargadas} imágenes.")
    print(f"Revisa la carpeta: {output_dir}")
    print("="*40)

if __name__ == "__main__":
    descargar_imagenes_ddg()