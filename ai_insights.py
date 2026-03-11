from google import genai
import pandas as pd

class AIAnalyzer:
    def generate_insights(self, api_key, df):
        try:
            # 1. Conexión exitosa comprobada
            client = genai.Client(api_key=api_key)
            
            resumen_estadistico = df.describe(include='all').to_string()
            columnas = ", ".join(df.columns.tolist())
            
            prompt = f"""
            Actúa como un Analista de Datos Senior.
            Estoy procesando un dataset con las siguientes variables: {columnas}.
            
            A continuación, te presento el resumen estadístico de los datos:
            {resumen_estadistico}
            
            Basado ESTRICTAMENTE en estos números, redacta:
            1. Tres (3) insights estadísticos clave o anomalías que detectes.
            2. Una (1) recomendación sobre qué variable debería investigarse más a fondo.
            
            Usa un tono profesional, directo y formateado con Markdown. No inventes datos.
            """
            
            # 2. Mecanismo de Cascada Anti-Errores 404
            # Ponemos todas las variantes de nombres posibles en los servidores de Google
            modelos_a_probar = [
                'gemini-2.5-flash',
                'gemini-2.0-flash',
                'gemini-1.5-flash-latest',
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro'
            ]
            
            ultimo_error = ""
            
            # 3. Prueba cada modelo hasta que uno funcione
            for modelo in modelos_a_probar:
                try:
                    response = client.models.generate_content(
                        model=modelo,
                        contents=prompt
                    )
                    return response.text # Si funciona, regresa el texto y se detiene
                except Exception as e:
                    ultimo_error = str(e)
                    continue # Si lanza 404, ignora y prueba el siguiente de la lista
            
            # Si terminan todos y ninguno funcionó (muy poco probable)
            return f"❌ Los servidores de Google rechazaron los modelos. Último error reportado: {ultimo_error}"
            
        except Exception as e:
            return f"❌ Error general en la configuración de la IA: {str(e)}"