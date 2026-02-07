import os
import requests
from duckduckgo_search import DDGS
from time import sleep

def descargar_imagenes_ddg():
    # Carpeta destino (usamos la misma estructura temporal)
    output_dir = os.path.join('datasets', 'analogos_raw')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    queries = {
        # ==========================================
        # CLASE CRÍTICA 1: VIRUS (Tuviste 0% acierto)
        # Estrategia: Usar Mosaicos de Tabaco y Rosa que son visualmente idénticos
        # ==========================================
        "virus_1": "blackberry yellow vein disease symptoms",
        "virus_2": "Rubus bushy dwarf virus leaf",
        "virus_3": "tobacco mosaic virus pattern leaf",      # ANÁLOGO (Textura mosaico)
        "virus_4": "rose mosaic virus yellow zigzag pattern", # ANÁLOGO (Patrón anillos)
        "virus_5": "cucumber mosaic virus leaf mottling",     # ANÁLOGO (Moteado)
        "virus_6": "blackberry chlorotic ringspot virus",

        # ==========================================
        # CLASE CRÍTICA 2: FUSARIUM / VERTICILLIUM (Marchitez)
        # Estrategia: Usar Tomate/Berenjena para necrosis vascular y hojas secas
        # ==========================================
        "fusarium_1": "fusarium wilt blackberry cane",
        "fusarium_2": "fusarium oxysporum tomato leaf wilt",  # ANÁLOGO (Hoja amarilla/seca)
        "fusarium_3": "vascular browning stem cross section", # Síntoma interno clave
        "verticillium_1": "verticillium wilt blackberry",
        "verticillium_2": "verticillium dahliae eggplant leaf", # ANÁLOGO (Necrosis en V)
        "verticillium_3": "raspberry cane blight symptoms",
        
        # ==========================================
        # CLASE CRÍTICA 3: BOTRYTIS (Confusión con Sano)
        # Estrategia: Fresa y Frambuesa para texturas de moho gris HD
        # ==========================================
        "botrytis_1": "botrytis cinerea blackberry fruit macro",
        "botrytis_2": "gray mold strawberry fruit close up",    # ANÁLOGO (Textura idéntica)
        "botrytis_3": "raspberry gray mold fungus",             # ANÁLOGO (Mismo fruto)
        "botrytis_4": "rotting blackberry gray fuzz",
        "botrytis_5": "botrytis sporulation texture",           # Solo textura

        # ==========================================
        # ROYA (RUST) - Refuerzo con Rosa
        # ==========================================
        "roya_1": "blackberry leaf rust symptoms",
        "roya_2": "Phragmidium violaceum blackberry",
        "roya_3": "rose rust leaf underside pustules",          # ANÁLOGO (Roya de la rosa)
        "roya_4": "orange rust raspberry leaf",                 # ANÁLOGO
        "roya_5": "gymnoconia nitens symptoms",

        # ==========================================
        # MILDEO (DOWNY/POWDERY) - Refuerzo con Vid/Calabaza
        # ==========================================
        "mildeo_1": "Peronospora sparsa blackberry",
        "mildeo_2": "grape downy mildew leaf underside",        # ANÁLOGO (Vid - Pelusa blanca)
        "mildeo_3": "powdery mildew zucchini leaf texture",     # ANÁLOGO (Calabaza - Polvo)
        "mildeo_4": "purple blotches blackberry leaf",
        "mildeo_5": "downy mildew rubus symptoms",

        # ==========================================
        # ANTRACNOSIS - Refuerzo con Vid
        # ==========================================
        "antracnosis_1": "anthracnose blackberry cane",
        "antracnosis_2": "grape anthracnose bird eye spot",     # ANÁLOGO (Ojo de pájaro)
        "antracnosis_3": "Elsinoe veneta blackberry lesions",
        "antracnosis_4": "bean anthracnose leaf veins",         # ANÁLOGO (Venas negras)

        # ==========================================
        # DAÑOS ABIÓTICOS (SOL/CLIMA)
        # ==========================================
        "dano_sol_1": "white drupelet disorder blackberry",
        "dano_sol_2": "sunscald blackberry fruit",
        "dano_sol_3": "UV damage raspberry fruit",              # ANÁLOGO
        "dano_rojo_4": "red cell regression blackberry",

        # ==========================================
        # SANO (HEALTHY) - Fundamental para evitar falsos positivos
        # ==========================================
        "sano_1": "healthy blackberry fruit on plant",
        "sano_2": "Rubus fruticosus healthy leaf",
        "sano_3": "fresh blackberry harvest macro",
        "sano_4": "green blackberry leaves texture",
        "sano_5": "thornless blackberry healthy cane"
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