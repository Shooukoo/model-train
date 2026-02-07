import cv2
import os
import glob

def recortador_de_texturas():
    # --- CONFIGURACIÓN ---
    input_folder = os.path.join('datasets', 'analogos_raw') 
    output_base = os.path.join('datasets', 'frutas', 'images', 'train')
    
    # Extensiones
    exts = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    # 1. ORDENAR ALFABÉTICAMENTE
    files.sort()
    
    print(f"--- SE ENCONTRARON {len(files)} IMÁGENES ---")
    
    # --- FUNCIÓN DE SALTO ---
    print("\n¿Quieres saltar fotos hasta llegar a un nombre específico?")
    print("Ejemplo: Si escribes 'verti', saltará todo hasta encontrar 'verticillium'.")
    start_word = input(">> Escribe nombre para empezar (o Enter para inicio): ").lower().strip()
    
    found_start = False 

    if not os.path.exists(input_folder):
        print(f"❌ Error: No existe la carpeta '{input_folder}'")
        return

    count = 0
    total_files = len(files)

    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        
        # --- LÓGICA DE SALTO ---
        if start_word and not found_start:
            if start_word not in filename.lower():
                continue
            else:
                found_start = True
                print(f"✅ ¡Encontrado! Empezando desde: {filename}")

        # --- CARGA DE IMAGEN ---
        img = cv2.imread(file_path)
        if img is None: continue

        # --- VISUALIZACIÓN ---
        scale_percent = 100
        display_img = img.copy()
        
        if img.shape[1] > 1200 or img.shape[0] > 800:
            scale_percent = 50 
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            display_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        while True:
            display_temp = display_img.copy()
            # Info en pantalla
            cv2.putText(display_temp, f"[{i+1}/{total_files}] {filename}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_temp, "Dibuja+Enter o solo Enter", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # --- SELECCIÓN ---
            r = cv2.selectROI("RECORTADOR", display_temp, showCrosshair=True)
            cv2.destroyAllWindows() 

            # --- PREGUNTA SI NO SE DIBUJÓ NADA ---
            if r[2] == 0 or r[3] == 0:
                print(f"\n[?] Foto: {filename}")
                print("   [Enter] = FOTO COMPLETA")
                print("   [0]     = SALTAR ESTA FOTO")
                print("   [Q]     = SALIR DEL PROGRAMA")
                decision = input("   >> ")
                
                if decision == '0':
                    print("   ⏭️ Saltada.")
                    break 
                elif decision.lower() == 'q':
                    print("   👋 Adiós.")
                    return 
                else:
                    imCrop = img # Original
            else:
                factor = 100 / scale_percent
                real_x = int(r[0] * factor)
                real_y = int(r[1] * factor)
                real_w = int(r[2] * factor)
                real_h = int(r[3] * factor)
                imCrop = img[real_y:real_y+real_h, real_x:real_x+real_w]

            # --- MENÚ ACTUALIZADO ---
            print(f"\n--- Clasificando: {filename} ---")
            print("1: Botrytis")
            print("2: Roya")
            print("3: Mildeo")
            print("4: Fusarium")
            print("5: Virus")
            print("6: Antracnosis")
            print("7: DANO (Sol)")
            print("8: SANO")
            print("9: VERTICILLIUM")  # <--- AGREGADO
            print("0: 🔙 REINTENTAR")
            
            opcion = input(">> Elige: ")

            cat_map = {
                '1': 'botrytis', 
                '2': 'roya', 
                '3': 'mildeo', 
                '4': 'fusarium', 
                '5': 'virus', 
                '6': 'antracnosis',
                '7': 'dano', 
                '8': 'sano',
                '9': 'verticillium' # <--- AGREGADO
            }

            if opcion in cat_map:
                categoria = cat_map[opcion]
                save_dir = os.path.join(output_base, categoria)
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                
                save_name = f"analog_{categoria}_{count}_{filename}"
                cv2.imwrite(os.path.join(save_dir, save_name), imCrop)
                print(f"   ✅ Guardado: {categoria}")
                count += 1
                break 
            elif opcion == '0':
                pass # Repite loop
            else:
                print("   ⚠️ Opción inválida.")

    print("\n¡Terminaste todas las fotos!")

if __name__ == "__main__":
    recortador_de_texturas()