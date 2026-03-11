from google import genai

# Reemplaza esto con tu llave real
MI_API_KEY = "AIzaSyAEHj5b2RETnvdYAB_MIt1OqAQlXA372o4" 

try:
    print("Conectando con Google AI Studio...")
    client = genai.Client(api_key=MI_API_KEY)
    
    print("\n✅ ¡Conexión exitosa! Tu cuenta tiene acceso a estos modelos:\n")
    
    # Pedimos la lista oficial de modelos a tu cuenta
    for model in client.models.list():
        # Filtramos para que solo muestre los que pueden generar texto
        if 'generateContent' in model.supported_generation_methods:
            print(f"- {model.name}")
            
except Exception as e:
    print(f"\n❌ Error al conectar: {e}")