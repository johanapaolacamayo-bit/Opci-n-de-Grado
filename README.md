# Agente de Apoyo Jurídico (prototipo académico)

Prototipo desarrollado para la monografía *"Fortalecimiento de la gestión jurídica
mediante inteligencia artificial generativa para la elaboración de comunicaciones
relacionadas con sucesiones y obligaciones crediticias en una entidad financiera"*
— Especialización en Inteligencia Artificial, UNIMINUTO.

## ¿Qué hace?

Es un agente de tipo **RAG (Retrieval-Augmented Generation)**: dado un repositorio
documental (normatividad pública + documentos institucionales simulados), permite
que un abogado escriba una consulta en lenguaje natural y reciba un borrador de
respuesta fundamentado en los documentos relevantes, junto con las fuentes usadas.

## Estructura del proyecto

```
agente_juridico/
├── data/                    # Repositorio documental (Tabla 4 de la monografía)
│   ├── 01_normatividad_sucesiones.txt
│   ├── 02_proteccion_datos.txt
│   ├── 03_politica_interna_simulada.txt
│   └── 04_formato_tipo_respuesta.txt
├── ingest.py                # Construye la base vectorial a partir de data/
├── agente.py                # Lógica de recuperación + generación (RAG)
├── app.py                   # Interfaz Streamlit
└── requirements.txt
```

## Cómo ejecutarlo

1. Crear un entorno virtual e instalar dependencias:
   ```
   python -m venv venv
   source venv/bin/activate      # en Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Construir el repositorio documental (solo una vez, o cuando cambien los archivos de `data/`):
   ```
   python ingest.py
   ```

3. Para que el agente genere texto real (y no solo muestre los fragmentos
   recuperados), tienes dos opciones:

   **Opción A — Ollama (gratuita, recomendada):**
   1. Descarga e instala Ollama desde https://ollama.com/download
   2. Descarga un modelo pequeño (una sola vez):
      ```
      ollama pull llama3.2
      ```
   3. Deja Ollama corriendo en segundo plano (normalmente se abre solo tras
      instalarlo) y simplemente ejecuta el agente; lo detecta automáticamente.

   **Opción B — API de Anthropic (de pago):**
   ```
   export ANTHROPIC_API_KEY="tu-llave-aqui"      # en Windows: $env:ANTHROPIC_API_KEY="tu-llave-aqui"
   ```

   Si no configuras ninguna de las dos, el agente sigue funcionando pero solo
   mostrará los fragmentos recuperados (útil para probar la recuperación sin
   generación de texto).

4. Probar por consola:
   ```
   python agente.py
   ```

5. Abrir la interfaz visual:
   ```
   streamlit run app.py
   ```

## Cómo usar esto en la monografía

- Los archivos de `data/` son la base de tu **Tabla 4** (repositorio documental).
- Las preguntas de prueba que definas (en `agente.py` o directamente en la interfaz)
  son la base de tu **Tabla 5** (casos de prueba).
- Ejecuta cada caso de prueba, observa la calidad de la respuesta y los fragmentos
  usados, y con eso completas tu **Tabla 6** y el capítulo de **Resultados**.
- Toma capturas de pantalla de la interfaz (`app.py`) para incluir como **Figura 6**
  ("Interacción del usuario con el agente de apoyo jurídico").

## Nota ética y de alcance

Todos los documentos de `data/` que no son normatividad pública están marcados
explícitamente como **simulados**, construidos únicamente con fines demostrativos
y metodológicos, dado que la investigación no tuvo acceso a información
confidencial de una entidad financiera real. Esto debe declararse también en el
documento de la monografía (apartado de Delimitación y Consideraciones Éticas).
