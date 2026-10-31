"""
app.py
------
Interfaz de chat en Streamlit para interactuar con el agente de apoyo jurídico.

Para ejecutar:
    streamlit run app.py

Requiere haber corrido antes: python ingest.py
"""

import streamlit as st
from agente import AgenteJuridico, MENSAJE_BIENVENIDA, DESCRIPCION_AGENTE

st.set_page_config(page_title="JoLex IA - Agente de Apoyo Jurídico", page_icon="⚖️")

st.title("⚖️ JoLex IA")
st.caption(DESCRIPCION_AGENTE)

# --- Estado de la sesión: el agente y el historial de la conversación ---
if "agente" not in st.session_state:
    with st.spinner("Cargando repositorio documental..."):
        st.session_state.agente = AgenteJuridico()

if "historial" not in st.session_state:
    # fuentes=None -> no se intentó búsqueda documental (no se muestra nada)
    st.session_state.historial = [
        {"role": "assistant", "content": MENSAJE_BIENVENIDA, "fuentes": None}
    ]


def mostrar_fuentes(fuentes):
    """Muestra las fuentes consultadas de forma legible y profesional.

    - fuentes is None -> no se intentó ninguna búsqueda (saludo, capacidades,
      orientación, etc.): no se muestra nada.
    - fuentes es una lista con contenido -> se muestra el listado.
    - fuentes es una lista vacía -> se intentó buscar pero no se encontró
      nada relevante: se indica explícitamente.
    """
    if fuentes is None:
        return
    # Mostrar las fuentes directamente y no dentro de un expander colapsado.
    # Así quedan visibles al validar y registrar los resultados de las pruebas.
    st.markdown("**Fuentes consultadas**")
    if fuentes:
        for etiqueta in fuentes:
            st.markdown(f"- {etiqueta}")
    else:
        st.markdown("No se encontraron documentos relacionados con la consulta realizada.")


# --- Mostrar el historial ya existente (como una conversación de chat) ---
for turno in st.session_state.historial:
    with st.chat_message(turno["role"]):
        st.write(turno["content"])
        mostrar_fuentes(turno.get("fuentes"))

# --- Cuadro de entrada tipo chat: se limpia solo después de enviar ---
pregunta = st.chat_input("Escribe tu consulta jurídica...")

if pregunta:
    # Muestra el mensaje del usuario inmediatamente
    st.session_state.historial.append({"role": "user", "content": pregunta, "fuentes": None})
    with st.chat_message("user"):
        st.write(pregunta)

    # Genera y muestra la respuesta del agente
    with st.chat_message("assistant"):
        with st.spinner("Consultando el repositorio y generando la respuesta..."):
            try:
                respuesta, fragmentos = st.session_state.agente.generar_respuesta(pregunta)
                fuentes = (
                    None if fragmentos is None
                    else st.session_state.agente.obtener_fuentes_amigables(fragmentos)
                )
            except Exception as exc:
                respuesta = (
                    "No fue posible procesar la consulta. Verifica que el repositorio "
                    "documental esté disponible y revisa el detalle técnico en la terminal."
                )
                fuentes = []
                print(f"Error al procesar la consulta JoLex IA: {exc}")
        st.write(respuesta)
        mostrar_fuentes(fuentes)

    st.session_state.historial.append({
        "role": "assistant",
        "content": respuesta,
        "fuentes": fuentes,
    })

# --- Barra lateral informativa ---
st.sidebar.header("Acerca de este prototipo")
st.sidebar.write(
    "Arquitectura: RAG (Retrieval-Augmented Generation)\n\n"
    "- Embeddings: sentence-transformers (local)\n"
    "- Base vectorial: ChromaDB (local)\n"
    "- Generación: modelo local vía Ollama (gratuito), con la API de "
    "Anthropic como alternativa opcional\n\n"
    "El repositorio documental usado combina normatividad pública y "
    "documentos institucionales simulados, dado que la investigación no "
    "tuvo acceso a información confidencial de una entidad real."
)

if st.sidebar.button("🗑️ Borrar conversación"):
    st.session_state.historial = [
        {"role": "assistant", "content": MENSAJE_BIENVENIDA, "fuentes": None}
    ]
    st.rerun()
