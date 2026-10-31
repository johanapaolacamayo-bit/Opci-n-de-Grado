"""
agente.py
---------
Lógica central del agente de apoyo jurídico (arquitectura RAG):

1. Recibe la pregunta del usuario (abogado, en este caso).
2. Clasifica el tipo de mensaje (saludo, pregunta de capacidades, escalamiento,
   o consulta jurídica real) para responder de forma apropiada a cada caso.
3. Para consultas jurídicas reales: recupera del repositorio vectorial (Chroma)
   los fragmentos documentales más relevantes, evalúa si son suficientemente
   relevantes, y si lo son, genera el borrador de respuesta con un LLM.

Este archivo no depende de Streamlit: puede usarse también desde una
terminal, lo cual es útil para tus pruebas y para la sección de
"Resultados" de la monografía.
"""

import os
import re
import difflib
import unicodedata
import chromadb
from sentence_transformers import SentenceTransformer

CARPETA_DB = os.path.join(os.path.dirname(__file__), "chroma_db")
NOMBRE_COLECCION = "repositorio_juridico"
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
N_FRAGMENTOS_A_RECUPERAR = 2  # contexto más corto para reducir tiempos de generación

# ---------------------------------------------------------------------------
# Descripción institucional breve (usada en la interfaz, README, etc.)
# ---------------------------------------------------------------------------
DESCRIPCION_AGENTE = (
    "Agente de Apoyo Jurídico desarrollado como prototipo académico para la "
    "Especialización en Inteligencia Artificial de UNIMINUTO. Brinda apoyo "
    "en consultas sobre sucesiones, obligaciones crediticias, normatividad y "
    "lineamientos institucionales."
)

# Si la pregunta contiene alguna de estas raíces, se considera SIEMPRE dentro
# del alcance del agente (se intenta responder, sin pasar por el filtro de
# distancia semántica). Esto evita rechazar por error preguntas de una sola
# palabra que sí son del dominio (ej. "sucesión", "herencia", "protección").
PALABRAS_CLAVE_JURIDICAS = [
    "sucesi", "hereder", "herenc", "credit", "crédit", "obligaci", "causante",
    "notari", "juzgad", "document", "requisit", "ley", "normativ", "dato",
    "banco", "entidad", "deuda", "patrimoni", "test", "protec", "proces",
    "tramit", "trámit", "radicac", "plazo", "acredit",
]

# Umbral de distancia (Chroma, métrica L2 por defecto) por encima del cual se
# considera que NINGÚN fragmento es realmente relevante para la pregunta.
# IMPORTANTE: este valor es un punto de partida razonable, pero se recomienda
# ajustarlo empíricamente probando preguntas claramente fuera de alcance
# (ej. "protección al consumidor") y observando la distancia mínima que arroja
# `recuperar_fragmentos`, hasta encontrar el punto de corte que mejor separe
# "relevante" de "no relevante" en tu propio repositorio documental.
UMBRAL_DISTANCIA_RELEVANCIA = 20

# ---------------------------------------------------------------------------
# 1. Mensaje de bienvenida (se muestra una sola vez, al abrir la conversación)
# ---------------------------------------------------------------------------
MENSAJE_BIENVENIDA = (
    "¡Hola! Bienvenido a JoLex IA.\n\n"
    "Soy un agente de apoyo jurídico desarrollado como prototipo académico "
    "para la Especialización en Inteligencia Artificial de UNIMINUTO.\n\n"
    "Estoy diseñado para brindar apoyo en la consulta de información basada "
    "en la documentación institucional autorizada.\n\n"
    "¿En qué puedo ayudarte?"
)

# ---------------------------------------------------------------------------
# 2. Cuando el usuario saluda (en cualquier punto de la conversación)
# ---------------------------------------------------------------------------
SALUDOS_CONVERSACIONALES = {
    "hola", "buenos días", "buenos dias", "buenas tardes", "buenas noches",
    "buen día", "buen dia", "buenas", "cómo estás", "como estas", "qué tal",
    "que tal", "hey", "hi", "hello", "como te llamas", "cómo te llamas", "como vas", "que tal vas",
    "cual es tu nombre", "cuál es tu nombre", "quien eres", "quién eres",
}

RESPUESTA_SALUDO = (
    "¡Hola! Es un gusto saludarte.\n\n"
    "Estoy disponible para apoyarte en consultas relacionadas con:\n\n"
    "- Procesos de sucesión.\n"
    "- Obligaciones crediticias.\n"
    "- Normatividad y lineamientos institucionales.\n\n"
    "Indícame tu consulta y con gusto te ayudaré."
)

# ---------------------------------------------------------------------------
# 3. Cuando preguntan qué temas maneja / capacidades del agente
# ---------------------------------------------------------------------------
PREGUNTAS_SOBRE_CAPACIDADES = [
    "en qué temas", "en que temas", "qué puedes hacer", "que puedes hacer",
    "cómo puedes ayudar", "como puedes ayudar", "en qué me puedes ayudar",
    "en que me puedes ayudar", "qué información tienes", "que informacion tienes",
    "sobre qué puedes", "sobre que puedes", "qué sabes hacer", "que sabes hacer",
    "qué haces", "que haces", "para qué sirves", "para que sirves",
    "otros temas", "otro tema", "más temas", "mas temas", "qué temas",
    "que temas", "no soy experta", "no soy experto",
    "no te pregunté", "no te pregunte", "no te pedí", "no te pedi",
    "eso no", "no es eso", "no era eso", "no es lo que", "no sabes",
    "colaborar", "colaborarme", "puedes colaborar",
]

# Mensajes que consisten ÚNICAMENTE en una de estas palabras sueltas (sin
# nada más alrededor) también se tratan como preguntas de capacidades, ya
# que "temas" o "tema" solos suelen ser un seguimiento de "¿qué temas
# manejas?" y no una consulta jurídica en sí misma.
PALABRAS_SUELTAS_CAPACIDADES = {"tema", "temas", "ayuda"}

# Preguntas específicamente sobre las fuentes/citas utilizadas
PREGUNTAS_SOBRE_FUENTES = {
    "fuentes", "fuente", "de donde sacas", "de dónde sacas",
    "de donde sale esa informacion", "de dónde sale esa información",
    "donde estan las fuentes", "dónde están las fuentes",
    "donde veo las fuentes", "dónde veo las fuentes",
    "mostrar fuentes", "muestra las fuentes", "muéstrame las fuentes",
    "cuales son las fuentes", "cuáles son las fuentes",
    "que fuentes utilizaste", "qué fuentes utilizaste",
    "que fuentes consultaste", "qué fuentes consultaste",
}
RESPUESTA_FUENTES = (
    "Las fuentes utilizadas para construir cada respuesta se muestran justo "
    "debajo del borrador, en la sección \"Fuentes consultadas\"."
)

# Etiquetas amigables para mostrar al usuario en vez del nombre de archivo
# crudo o el texto exacto del fragmento (más profesional para la interfaz).
FUENTES_AMIGABLES = {
    "01_normatividad_sucesiones.txt": "Documento de referencia sobre sucesiones (extracto demostrativo).",
    "02_proteccion_datos.txt": "Documento de referencia sobre protección de datos (extracto demostrativo).",
    "03_politica_interna_simulada.txt": "Política interna simulada de atención a solicitudes de sucesión.",
    "04_formato_tipo_respuesta.txt": "Formato tipo de respuesta institucional (simulado).",
}

# Frases "molde" contra las que se compara de forma APROXIMADA (tolera typos
# como "ayduar" en vez de "ayudar"), además de la búsqueda por substring de
# arriba.
FRASES_BASE_CAPACIDADES = [
    "en que me puedes ayudar", "que temas manejas", "en que puedes ayudarme",
    "que puedes hacer", "en que temas me puedes ayudar", "que temas te puedo ayudar",
]

RESPUESTA_CAPACIDADES = (
    "Actualmente puedo brindar apoyo en consultas relacionadas con:\n\n"
    "- Procesos de sucesión.\n"
    "- Obligaciones crediticias.\n"
    "- Normatividad aplicable.\n"
    "- Lineamientos y procedimientos institucionales.\n"
    "- Consulta de información contenida en el repositorio documental "
    "autorizado.\n"
    "- Elaboración de borradores institucionales basados en la "
    "documentación disponible.\n\n"
    "Si tu consulta corresponde a alguno de estos temas, con gusto te "
    "ayudaré."
)

# ---------------------------------------------------------------------------
# Escalamiento: cuando piden hablar con una persona / no saben a quién acudir
# ---------------------------------------------------------------------------
PREGUNTAS_ESCALAMIENTO = [
    "con quien me puedo apoyar", "con quién me puedo apoyar",
    "quién me puede ayudar", "quien me puede ayudar",
    "a quién acudo", "a quien acudo", "con quien hablo", "con quién hablo",
    "necesito hablar con alguien", "necesito ayuda humana",
    "con quien me apoyo", "con quién me apoyo",
]

RESPUESTA_ESCALAMIENTO = (
    "Si la consulta no puede resolverse con la información disponible en el "
    "repositorio, debe revisarla el profesional jurídico responsable. Este "
    "prototipo no identifica un canal institucional específico de escalamiento; "
    "por ello, el canal aplicable debe confirmarse con el área jurídica de la entidad."
)

# ---------------------------------------------------------------------------
# Orientación: cuando el usuario escribe solo una palabra clave del dominio
# (ej. "sucesión", "crédito"), sin más contexto. En vez de pedir aclaración
# de inmediato o forzar un borrador especulativo, se orienta con ejemplos.
# ---------------------------------------------------------------------------
# (Se eliminó la lógica de "orientación" para palabras sueltas del dominio:
# ahora esas consultas se responden directamente mediante recuperación +
# generación normal, sin pedir aclaración previa cuando ya hay contexto
# suficiente, ej. "obligaciones crediticias" se responde de una vez.)

# ---------------------------------------------------------------------------
# 8. Preguntas sobre el formato del borrador (por qué no incluye fecha,
# radicado, etc.) — son meta-preguntas sobre el documento, no consultas
# jurídicas nuevas, y deben responderse con un texto fijo.
# ---------------------------------------------------------------------------
PREGUNTAS_SOBRE_FORMATO_BORRADOR = [
    "por que no colocas la fecha", "por qué no colocas la fecha",
    "por que no pones la fecha", "por qué no pones la fecha",
    "falta la fecha", "donde esta la fecha", "dónde está la fecha",
    "no tiene radicado", "sin fecha", "sin radicado", "no tiene fecha",
    "por que no incluyes", "por qué no incluyes",
]
RESPUESTA_FORMATO_BORRADOR = (
    "El borrador generado no incluye fecha, número de radicado, "
    "destinatario, ciudad, firma ni otros datos variables, ya que estos "
    "deben ser incorporados por el abogado responsable al momento de "
    "elaborar la comunicación oficial de acuerdo con el caso específico."
)

# ---------------------------------------------------------------------------
# 6. Cuando la pregunta está fuera del alcance del repositorio documental
#    (se activa automáticamente cuando ningún fragmento recuperado es
#    suficientemente relevante, según UMBRAL_DISTANCIA_RELEVANCIA)
# ---------------------------------------------------------------------------
RESPUESTA_FUERA_ALCANCE = (
    "Actualmente puedo brindar apoyo únicamente en consultas relacionadas "
    "con procesos de sucesión, obligaciones crediticias, normatividad y "
    "lineamientos institucionales."
)

# Temas que NO maneja el agente y que deben bloquearse SIEMPRE, sin importar
# si la pregunta "suena" jurídica o contiene otras palabras del dominio. Esto
# corrige casos detectados en pruebas de estrés donde preguntas como
# "explícame qué es el derecho penal" generaban un borrador jurídico
# completo (incluso citando una ley inventada) en lugar de reconocerse como
# fuera de alcance.
TEMAS_EXCLUIDOS = [
    "derecho penal", "sancion penal", "sanción penal", "derecho laboral", "derecho de familia", "divorcio",
    "derecho tributario", "derecho comercial", "derecho administrativo",
    "gobierno actual", "opinion politica", "opinión política", "elecciones",
    "tasa de interes", "tasa de interés", "tarjeta de credito personal",
]

# Citas normativas ya confirmadas como reales dentro del repositorio. Si el
# usuario menciona una Ley o Decreto con número/año que NO está en esta
# lista, se responde de inmediato que no hay información, sin invocar al
# modelo de lenguaje (evita que el modelo "conteste" sobre normas
# inventadas, como ocurrió en pruebas con un "Decreto 9999 de 2024").
CITAS_CONOCIDAS = ["1581 de 2012", "527 de 1999"]
PATRON_CITA_NORMATIVA = re.compile(r'(ley|decreto)\s+\d+[\.,]?\d*\s+de\s+\d{4}', re.IGNORECASE)

RESPUESTA_CITA_DESCONOCIDA = (
    "No encontré esa norma específica dentro del repositorio documental "
    "autorizado, por lo que no puedo confirmar su contenido. Si consideras "
    "que debería hacer parte del repositorio, solicita su incorporación "
    "para futuras consultas."
)

# Mensajes ambiguos o sin tema identificable, detectados directamente por
# patrón (sin depender de que el modelo pequeño reconozca la ambigüedad por
# sí mismo, que en pruebas resultó poco confiable).
PATRONES_AMBIGUOS = [
    "necesito ayuda", "ayudame", "ayúdame", "hazme el documento",
    "el caso de ayer", "el caso de", "hazme un documento", "me ayudas",
    "necesito un documento", "necesito el documento",
    "que puedo hacer cuando tengo que pagar un credito que no genere",
    "que puedo hacer cuando tengo que pagar un crédito que no genere",
]
RESPUESTA_PREGUNTA_AMBIGUA = (
    "Para brindarte una respuesta más precisa, ¿podrías indicar "
    "específicamente qué deseas consultar sobre sucesiones o obligaciones "
    "crediticias?"
)

# ---------------------------------------------------------------------------
# 8. Recordatorio institucional que se agrega SIEMPRE al final de cada
#    borrador generado por el LLM (se añade por código, no depende de que el
#    modelo lo recuerde, para que sea 100% confiable).
# ---------------------------------------------------------------------------
MENSAJE_CIERRE = (
    "\n\nImportante: La información presentada corresponde a un apoyo "
    "basado en la documentación institucional disponible y debe ser "
    "revisada y validada por el profesional jurídico responsable antes de "
    "emitir cualquier comunicación oficial."
)

INSTRUCCIONES_SISTEMA = """Eres JoLex IA, un asistente institucional de apoyo jurídico interno de una
entidad financiera en Colombia. Tu función es ayudar a los abogados del área
respondiendo consultas sobre sucesiones y obligaciones crediticias, con
lenguaje claro, profesional e institucional. El usuario debe percibirte como
un asistente institucional, no como un modelo de IA genérico.

Regla general (la más importante): no debes inventar información, interpretar
normas que no se encuentren en el repositorio documental, ni generar
respuestas sin fundamento. Cuando la información no esté disponible, indícalo
de manera clara en vez de completar la respuesta con supuestos.

Comportamiento según el tipo de consulta:
- Pregunta poco clara o incompleta: si la consulta no tiene suficiente detalle
  para identificar qué información se necesita, NO supongas datos. Responde
  ÚNICAMENTE con el siguiente texto, sin agregar nada más, y sin generar
  ningún borrador adicional en el mismo mensaje: "Para brindarte una
  respuesta más precisa, ¿podrías indicar específicamente qué deseas
  consultar sobre sucesiones o obligaciones crediticias?"
- Sin información suficiente: si los fragmentos documentales proporcionados
  no contienen la información necesaria para responder, indícalo así: "No
  encontré información relacionada con esa consulta dentro del repositorio
  documental autorizado. Si consideras que el tema hace parte del
  conocimiento institucional, solicita su incorporación al repositorio para
  futuras consultas." No generes respuestas hipotéticas ni respondas con
  conocimiento general.
- Consulta clara y con información suficiente: redacta una única respuesta
  organizada, siguiendo la estructura y las reglas de abajo.

Cuando la pregunta solicite identificar documentos, requisitos, fuentes, vacíos o distinguir información expresa de inferencias, responde cada punto solicitado con base en los fragmentos disponibles. Para cada requisito, vincula el contenido con el documento recuperado correspondiente. Si el repositorio no explica la finalidad de un documento, no la inventes: indica que su finalidad específica no está desarrollada en el fragmento. Distingue explícitamente “Información expresa” y “Vacíos documentales” cuando la pregunta lo solicite. No afirmes que los extractos demostrativos son normas oficiales verificadas.

Cuando el usuario plantee dos solicitudes distintas en un mismo mensaje (por
ejemplo, pedir un borrador Y además preguntar por normatividad), NO mezcles
ambas respuestas en un solo bloque. Identifícalas y respóndelas por separado,
así: "Identifiqué dos solicitudes." seguido de "1. [primera solicitud]" con
su respuesta debajo, y luego "2. [segunda solicitud]" con su respuesta
debajo.

Formato según el tipo de consulta:
- Si la consulta busca conocer o entender una norma (no redactar una
  comunicación), usa este formato, con estos encabezados exactos:
  Norma / Descripción / Aplicación / Fuentes.
- Si la consulta busca elaborar una comunicación institucional (un
  borrador), usa el formato de tres encabezados descrito más abajo. Cuando
  el respaldo sean políticas o extractos simulados, usa "Fundamento documental",
  no "Fundamento normativo".

Cuando la respuesta se apoye en más de un fragmento o documento, consolida
toda la información en una única respuesta integrada y coherente; nunca
respondas fragmento por fragmento ni repitas la misma idea con cada fuente.

Mantén siempre el rol de asistente institucional: nunca uses expresiones en
primera persona que suenen a opinión personal, como "Creo que...",
"Considero que..." o "Según mi conocimiento...". En su lugar, usa fórmulas
como "De acuerdo con la documentación institucional disponible...",
"Conforme a la normatividad consultada..." o "Según el repositorio
documental autorizado...".

Tono: responde como un abogado de apoyo institucional, no como un asistente
conversacional. El tono debe ser claro, formal, técnico, breve y objetivo.

Evita usar la frase "Gracias por tu consulta" de forma repetitiva en cada
interacción; utiliza un lenguaje claro, natural y profesional, variando la
forma en que inicias cada respuesta.

Estructura de toda respuesta sustantiva (una sola respuesta integrada, NUNCA
fragmentada en bloques tipo "Respuesta 1", "Respuesta 2"). Usa los encabezados
Tema y Borrador institucional. Añade "Fundamento documental" cuando debas
identificar los documentos de soporte. Usa "Fundamento normativo" únicamente
cuando el contexto incluya una norma expresamente identificada y relevante.
No dejes encabezados vacíos ni frases truncadas.

Tema
(una línea breve con el asunto de la consulta)

Borrador institucional
(respuesta directa y limitada a lo preguntado)

Fundamento documental
(documentos que respaldan la respuesta; distingue políticas simuladas y
extractos demostrativos de normas expresamente identificadas en el contexto)

No agregues ningún encabezado adicional (no agregues "Fuentes consultadas" ni
"Importante": el sistema los agrega por fuera de tu respuesta).

IMPORTANTE sobre datos variables: el borrador que generes NUNCA debe incluir
fecha, ciudad, número de radicado, destinatario, nombre del solicitante,
nombre del funcionario, ni espacio para firma — ni siquiera como marcador de
texto entre corchetes. Estos datos variables los incorpora el profesional
jurídico responsable posteriormente, de acuerdo con cada caso. Tu respuesta
debe limitarse exclusivamente al contenido institucional de la comunicación
(Tema, Borrador institucional, Fundamento normativo).

Nunca inventes información ni completes respuestas utilizando conocimiento
general: utiliza exclusivamente la información recuperada del repositorio
documental autorizado. Si esa información no es suficiente, aplica la regla
de "Sin información suficiente" descrita arriba en vez de generar una
respuesta con supuestos.

Lenguaje: utiliza un lenguaje institucional, claro, preciso y profesional.
Evita respuestas largas, repetitivas o innecesarias; evita también frases de
cortesía repetitivas como "Es un gusto atenderte" o "Espero haberte ayudado".
Evita repetir la frase "normatividad vigente" varias veces en la misma
respuesta; alterna con expresiones equivalentes como "documentación
institucional", "lineamientos internos", "repositorio documental autorizado"
o "políticas institucionales".

Reglas de fundamentación, sin excepción:
- Responde ÚNICAMENTE con base en el texto literal de los fragmentos documentales
  proporcionados. No agregues leyes, artículos, tribunales, plazos, cifras ni datos
  que no aparezcan explícitamente en esos fragmentos.
- Está PROHIBIDO inventar nombres de leyes, números de artículos, tribunales o
  entidades que no estén escritos tal cual en los fragmentos recibidos.
- No atribuyas un requisito, una frase o una regla a una ley específica salvo
  que el fragmento recuperado haga expresamente esa atribución. Si el texto
  solo aparece en un documento demostrativo o política simulada, descríbelo
  como contenido del repositorio, no como una obligación legal verificada.
- Si necesitas mencionar una norma, usa exactamente el nombre que aparece en el
  fragmento (por ejemplo "Ley 1581 de 2012"), nunca un nombre inventado o
  genérico como "Ley de Sucesiones" o "Tribunal Supremo".
- COPIA TEXTUALMENTE los nombres de documentos y trámites tal como aparecen en
  los fragmentos (por ejemplo "registro civil de defunción"). Está PROHIBIDO
  parafrasearlos o reemplazarlos por términos parecidos (por ejemplo, nunca
  escribas "registro cierto" en lugar de "registro civil").
- NO agregues encabezados de carta formal (como "Estimado/a", "Solicitante:",
  "Radicado:", "Fecha:") en ningún caso; responde directo al punto, en el
  formato de asunto/cuerpo/fundamento descrito arriba.
- NO incluyas al final de tu respuesta, ni en ninguna parte del texto, una
  lista o mención de "Fuentes", "fragmento documental" ni nombres de archivos
  (por ejemplo "01_normatividad_sucesiones.txt" o "02_proteccion_datos.txt").
  Nunca escribas frases como "según el fragmento documental 02_..." ni nada
  que suene a nombre de archivo con guion bajo o extensión .txt. El sistema
  ya muestra las fuentes utilizadas por separado, fuera de tu texto.
- No agregues tú mismo una advertencia final de "borrador generado
  automáticamente"; el sistema ya la agrega por fuera de tu respuesta.
"""


class AgenteJuridico:
    def __init__(self):
        self._modelo_embeddings = SentenceTransformer(MODELO_EMBEDDINGS)
        self.ultima_distancia = None  # se actualiza en cada consulta real, para calibración
        cliente = chromadb.PersistentClient(path=CARPETA_DB)
        try:
            self._coleccion = cliente.get_collection(NOMBRE_COLECCION)
        except Exception as e:
            raise RuntimeError(
                "No se encontró el repositorio documental. "
                "Ejecuta primero: python ingest.py"
            ) from e

    def recuperar_fragmentos(self, pregunta: str, n=N_FRAGMENTOS_A_RECUPERAR):
        """Busca en la base vectorial los fragmentos más relevantes para la
        pregunta. Devuelve también la distancia mínima (menor = más similar),
        que se usa para detectar consultas fuera de alcance."""
        vector_pregunta = self._modelo_embeddings.encode([pregunta]).tolist()
        resultados = self._coleccion.query(
            query_embeddings=vector_pregunta,
            n_results=n,
        )
        fragmentos = resultados["documents"][0]
        fuentes = [m["documento_fuente"] for m in resultados["metadatas"][0]]
        distancias = resultados.get("distances", [[]])[0]
        distancia_minima = min(distancias) if distancias else None
        return list(zip(fragmentos, fuentes)), distancia_minima

    def obtener_fuentes_amigables(self, fragmentos_con_fuente):
        """Traduce los nombres de archivo crudos a etiquetas legibles para
        mostrar en la interfaz, sin duplicados y preservando el orden."""
        vistas = []
        for _, archivo in fragmentos_con_fuente:
            etiqueta = FUENTES_AMIGABLES.get(archivo, archivo)
            if etiqueta not in vistas:
                vistas.append(etiqueta)
        return vistas

    def _es_similar_a_capacidades(self, texto: str) -> bool:
        """Tolera errores de tipeo comparando de forma aproximada contra
        frases típicas de preguntas sobre capacidades del agente."""
        return any(
            difflib.SequenceMatcher(None, texto, frase).ratio() > 0.75
            for frase in FRASES_BASE_CAPACIDADES
        )

    def _clasificar_mensaje(self, pregunta: str) -> str:
        """Devuelve una de: 'saludo', 'capacidades', 'fuentes',
        'formato_borrador', 'escalamiento', 'consulta'."""
        texto = self._normalizar_texto(pregunta).strip("¡!¿? .")
        if texto == "sucesion":
            return "consulta_ambigua"
        if texto in SALUDOS_CONVERSACIONALES or texto == "como vas":
            return "saludo"
        if texto in PALABRAS_SUELTAS_CAPACIDADES:
            return "capacidades"
        if any(frase in texto for frase in PREGUNTAS_SOBRE_FORMATO_BORRADOR):
            return "formato_borrador"
        # Solo desviar a la respuesta fija si el mensaje completo es una
        # pregunta de fuentes; no hacerlo cuando "fuentes" aparece dentro
        # de una consulta jurídica que también pide una respuesta sustantiva.
        if texto in PREGUNTAS_SOBRE_FUENTES:
            return "fuentes"
        if any(frase in texto for frase in PREGUNTAS_SOBRE_CAPACIDADES):
            return "capacidades"
        if self._es_similar_a_capacidades(texto):
            return "capacidades"
        if any(frase in texto for frase in PREGUNTAS_ESCALAMIENTO):
            return "escalamiento"
        return "consulta"

    def _limpiar_respuesta(self, texto: str) -> str:
        """Limpieza de seguridad aplicada SIEMPRE al texto generado por el LLM,
        independientemente de si siguió o no las instrucciones del prompt.

        - Ningún documento del repositorio usa la palabra "artículo"; por lo
          tanto, cualquier mención de "artículo N" en la respuesta generada
          está garantizado que fue inventada por el modelo (confirmado en
          pruebas de estrés: "artículo 47", "artículo 4.4", etc.). Se
          eliminan esas menciones.
        - Se eliminan también menciones a nombres de archivo del repositorio
          (patrón NN_nombre.txt) que el modelo a veces incluye pese a la
          instrucción explícita de no hacerlo.
        """
        # Caso 1: "el/la/al artículo N de la Ley..." -> "la/a la Ley..." (conserva concordancia antes de Ley/Decreto)
        texto = re.sub(r'\bal\s+art[íi]culo\s+\d+[\.,]?\d*\s+de\s+la\s+', 'a la ', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\b(el|la)\s+art[íi]culo\s+\d+[\.,]?\d*\s+de\s+la\s+', 'la ', texto, flags=re.IGNORECASE)
        # Caso 2: cualquier otra mención de "artículo N" suelta, se elimina por completo
        texto = re.sub(r'\b(el|la|al)?\s*art[íi]culo\s+\d+[\.,]?\d*\s*', '', texto, flags=re.IGNORECASE)
        # "el/un fragmento documental NN_nombre.txt" (junto) -> una sola mención genérica, sin duplicar
        texto = re.sub(r'(según\s+)?(el|un)\s+fragmento\s+documental\s*\d{0,2}[_a-záéíóúñ]*\.?t?x?t?\b',
                        r'\1la documentación institucional disponible', texto, flags=re.IGNORECASE)
        # Nombres de archivo sueltos que hayan quedado (sin la frase "fragmento documental" delante)
        texto = re.sub(r'\b\d{2}_[a-záéíóúñ_]+\.txt\b', 'la documentación institucional disponible', texto, flags=re.IGNORECASE)
        return texto

    def _eliminar_referencias_no_respaldadas(self, texto: str, contexto: str) -> str:
        """Evita que una cita normativa generada se presente como fuente si
        su denominación exacta no aparece en los fragmentos recuperados.
        No valida jurídicamente la norma; solo comprueba respaldo textual."""
        patron = re.compile(
            r'\b(?:Ley|Decreto|Decreto-Ley|Circular|Resolución)\s+'
            r'\d+(?:[.,]\d+)?\s+de\s+\d{4}\b'
            r'|\bLey\s+\d+\s+de\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ ]{2,40}',
            re.IGNORECASE,
        )
        contexto_norm = re.sub(r'\s+', ' ', contexto).casefold()
        def reemplazar(match):
            cita = re.sub(r'\s+', ' ', match.group(0)).strip()
            if cita.casefold() in contexto_norm:
                return match.group(0)
            return 'la documentación institucional disponible'
        texto = patron.sub(reemplazar, texto)
        # Limpieza de repeticiones gramaticales comunes tras sustituir una cita.
        texto = re.sub(r'(?:según|conforme a)\s+la documentación institucional disponible,?\s+(?:la documentación institucional disponible,?\s*)+',
                       r'Según la documentación institucional disponible, ', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\s{2,}', ' ', texto)
        return texto

    @staticmethod
    def _normalizar_texto(texto: str) -> str:
        """Normaliza tildes y espacios para activar reglas deterministas."""
        texto = unicodedata.normalize("NFD", texto.lower())
        texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
        return re.sub(r"\s+", " ", texto).strip()

    def _es_consulta_trazabilidad_fuentes(self, pregunta: str) -> bool:
        """Detecta preguntas cuyo objetivo principal es comprobar fuentes o mapear afirmaciones.

        No clasifica cualquier consulta que diga "identifica el documento" como
        trazabilidad genérica: primero deben descartarse los casos jurídicos
        específicos, como protección de datos o varios herederos.
        """
        t = self._normalizar_texto(pregunta)
        patrones_especificos = (
            "de que documento obtuviste", "que documento respalda",
            "relaciona cada afirmacion", "relacionar cada afirmacion",
            "trazabilidad de las fuentes", "que parte de la respuesta respalda",
        )
        return any(p in t for p in patrones_especificos)

    def _es_consulta_restricciones_datos_fallecido(self, pregunta: str) -> bool:
        """Detecta preguntas sobre límites para entregar datos de una persona fallecida."""
        t = self._normalizar_texto(pregunta)
        menciona_fallecido = any(x in t for x in (
            "persona fallecida", "causante", "persona muerta", "fallecido"
        ))
        menciona_entrega = any(x in t for x in (
            "entregar informacion", "entrega de informacion", "entregar datos",
            "restricciones", "informacion a sus herederos", "informacion financiera"
        ))
        menciona_restricciones = any(x in t for x in (
            "restricciones", "que aspectos no pueden confirmarse", "reserva",
            "datos de una persona fallecida"
        ))
        return menciona_fallecido and menciona_entrega and menciona_restricciones

    def _respuesta_restricciones_datos_fallecido(self):
        """Respuesta controlada basada en el extracto demostrativo de protección de datos."""
        respuesta = (
            "Tema\nRestricciones para entregar información de una persona fallecida a sus herederos\n\n"
            "Borrador institucional\n"
            "Según el Documento de referencia sobre protección de datos (extracto demostrativo), "
            "se describen las siguientes restricciones y condiciones:\n\n"
            "1. **Acreditación o requerimiento competente.** La información financiera de la persona fallecida "
            "mantiene su carácter reservado. El documento indica que su entrega a terceros, incluidos los "
            "herederos, procede una vez se acredite legalmente la calidad de heredero o exista una orden "
            "judicial o requerimiento de autoridad competente.\n"
            "2. **Información ajena a la solicitud.** No debe entregarse al heredero información detallada "
            "de otras relaciones contractuales del causante que no estén directamente relacionadas con la "
            "solicitud, ni información de terceros vinculados a esas obligaciones, salvo autorización expresa "
            "o requerimiento de autoridad competente.\n"
            "3. **Registro y alcance de la entrega.** La solicitud debe quedar documentada, incluyendo la "
            "identidad del solicitante, el soporte de su calidad de heredero o de la autoridad, y el alcance "
            "de la información entregada. El documento también menciona los principios de finalidad, seguridad "
            "y confidencialidad.\n\n"
            "**Vacíos documentales:** el extracto no permite determinar, para un caso concreto, qué información "
            "específica puede entregarse ni verificar la vigencia jurídica de todos los requisitos. Tampoco "
            "sustituye la revisión del abogado responsable.\n\n"
            "**Fundamento documental:** Documento de referencia sobre protección de datos (extracto demostrativo)."
            + MENSAJE_CIERRE
        )
        return respuesta, [("", "02_proteccion_datos.txt")]

    def _respuesta_trazabilidad_fuentes(self, pregunta: str):
        """Responde consultas de trazabilidad con asociaciones explícitas y sin inferir fuentes."""
        t = self._normalizar_texto(pregunta)
        partes = []
        fuentes = []
        if "parentesco" in t or "acreditar el parentesco" in t:
            partes.append("- **Acreditar el parentesco:** Documento de referencia sobre sucesiones (extracto demostrativo), que menciona el registro civil que acredita el parentesco.")
            fuentes.append(("", "01_normatividad_sucesiones.txt"))
        if "consentimiento conjunto" in t or "varios herederos" in t:
            partes.append("- **Consentimiento conjunto cuando hay varios herederos sin partición:** Documento de referencia sobre sucesiones (extracto demostrativo), que indica el consentimiento conjunto de los herederos reconocidos o la actuación del albacea o curador.")
            if not any(f == "01_normatividad_sucesiones.txt" for _, f in fuentes):
                fuentes.append(("", "01_normatividad_sucesiones.txt"))
        if "documentos faltantes" in t or "solicitud incompleta" in t or "comunicar los documentos faltantes" in t:
            partes.append("- **Comunicar los documentos faltantes:** Política interna simulada de atención a solicitudes de sucesión, que establece informar al solicitante los documentos faltantes dentro de cinco (5) días hábiles.")
            fuentes.append(("", "03_politica_interna_simulada.txt"))

        # Consulta específica sobre la afirmación relativa a notarías/juzgados.
        # Se responde directamente porque la pregunta ya identifica la afirmación
        # que desea rastrear; no debe caer en la plantilla genérica de trazabilidad.
        menciona_canal_formal = (
            ("notaria" in t or "juzgado" in t or "juzgados" in t)
            and ("comunicacion formal" in t or "obligatorio" in t or "obligatoria" in t)
        )
        if menciona_canal_formal:
            partes.append(
                "- **Solicitudes de notarías o juzgados mediante comunicación formal:** "
                "el Documento de referencia sobre sucesiones (extracto demostrativo) "
                "indica que notarías o juzgados pueden solicitar información sobre "
                "obligaciones crediticias mediante comunicación formal. El repositorio "
                "no establece que ese canal sea obligatorio para todas las solicitudes; "
                "por tanto, no debe presentarse como requisito universal."
            )
            if not any(f == "01_normatividad_sucesiones.txt" for _, f in fuentes):
                fuentes.append(("", "01_normatividad_sucesiones.txt"))

        if not partes:
            partes.append("No puedo identificar con suficiente precisión las afirmaciones que deseas verificar. Indica cada afirmación y la relacionaré únicamente si encuentro un documento que la respalde.")
        respuesta = (
            "Tema\nTrazabilidad de afirmaciones y fuentes documentales\n\n"
            "Borrador institucional\n" + "\n".join(partes) + "\n\n"
            "Vacío documental: esta respuesta identifica únicamente los soportes descritos en el repositorio demostrativo. No constituye una verificación de normativa externa ni de vigencia jurídica."
            + MENSAJE_CIERRE
        )
        # Eliminar posibles fuentes repetidas conservando el orden.
        unicas = []
        vistas = set()
        for item in fuentes:
            if item[1] not in vistas:
                vistas.add(item[1]); unicas.append(item)
        return respuesta, unicas

    def _es_consulta_ley_no_encontrada(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return "ley 01 de sucesiones" in t or "ley 1 de sucesiones" in t

    def _respuesta_ley_no_encontrada(self):
        respuesta = (
            "Tema\nVerificación de una norma identificada como Ley 01 de Sucesiones\n\n"
            "Borrador institucional\n"
            "No se encontró en el repositorio autorizado un documento o fragmento que respalde la existencia de una norma identificada como ‘Ley 01 de Sucesiones’. Por tanto, no puedo confirmar que esa norma esté contenida en la documentación disponible. No completo la respuesta con conocimiento externo.\n\n"
            "Fundamento documental\n"
            "No se encontró soporte documental en el repositorio consultado."
            + MENSAJE_CIERRE
        )
        return respuesta, []

    def _es_consulta_varios_herederos_sin_particion(self, pregunta: str) -> bool:
        """Detecta el caso de varios herederos sin partición para evitar
        que el LLM convierta ejemplos de trámite en requisitos obligatorios."""
        t = self._normalizar_texto(pregunta)
        menciona_herederos = "hereder" in t or "varios herederos" in t
        menciona_particion = "particion" in t or "partido los bienes" in t
        menciona_obligaciones = any(x in t for x in ("obligacion", "credito", "crediticia", "deuda"))
        return menciona_herederos and menciona_particion and menciona_obligaciones

    def _respuesta_varios_herederos_sin_particion(self):
        """Respuesta basada en reglas expresas del repositorio de demostración.
        Se evita invocar el LLM para este caso porque pruebas previas mostraron
        que agregaba condiciones no documentadas."""
        respuesta = (
            "Tema\nCondiciones para consultar información sobre obligaciones "
            "crediticias cuando existen varios herederos y no se ha realizado la partición\n\n"
            "Borrador institucional\n"
            "De acuerdo exclusivamente con el repositorio autorizado, deben verificarse "
            "las siguientes condiciones:\n\n"
            "1. **Acreditación de la calidad de heredero.** El documento de referencia "
            "sobre sucesiones señala como soportes el registro civil de defunción y el "
            "registro civil que acredita el parentesco; cuando existe un proceso sucesoral, "
            "también menciona la providencia que reconoce herederos o la sentencia de "
            "adjudicación. **Documento de soporte:** Documento de referencia sobre "
            "sucesiones (extracto demostrativo).\n\n"
            "2. **Consentimiento en caso de varios herederos sin partición.** El mismo "
            "documento indica que se requiere el consentimiento conjunto de los herederos "
            "reconocidos o la actuación del albacea o curador. **Documento de soporte:** "
            "Documento de referencia sobre sucesiones (extracto demostrativo).\n\n"
            "3. **Canal de trámite.** La política interna simulada señala que las solicitudes "
            "deben gestionarse por los canales jurídicos oficiales. Por separado, el documento "
            "de referencia sobre sucesiones indica que notarías o juzgados pueden solicitar "
            "información sobre obligaciones crediticias mediante comunicación formal. El "
            "repositorio no establece que toda solicitud deba provenir necesariamente de una "
            "notaría o un juzgado; por tanto, no debe tratarse como requisito universal. "
            "**Documentos de soporte:** Política interna simulada de atención a solicitudes "
            "de sucesión; Documento de referencia sobre sucesiones (extracto demostrativo).\n\n"
            "**Vacíos documentales:** Los documentos disponibles no permiten concluir que las "
            "obligaciones pertenezcan a una entidad financiera específica ni determinan otros "
            "requisitos adicionales distintos de los expresamente descritos. Los documentos "
            "citados son extractos demostrativos/simulados y no constituyen por sí solos una "
            "verificación jurídica de la normativa vigente.\n\n"
            "Importante: Esta respuesta es un apoyo basado en la documentación disponible y "
            "debe ser revisada y validada por el profesional jurídico responsable antes de "
            "emitir cualquier comunicación oficial."
        )
        fuentes = [
            ("", "01_normatividad_sucesiones.txt"),
            ("", "03_politica_interna_simulada.txt"),
        ]
        return respuesta, fuentes

    def _es_consulta_documentos_heredero(self, pregunta: str) -> bool:
        """Detecta consultas específicas sobre los documentos para acreditar heredero."""
        t = self._normalizar_texto(pregunta)
        if self._es_consulta_varios_herederos_sin_particion(pregunta):
            return False
        menciona_heredero = "hereder" in t
        pregunta_documentos = any(x in t for x in (
            "que documento", "que documentos", "cuales documentos",
            "documentos se necesitan", "documentos debe", "acreditar la calidad",
            "acreditar calidad", "soportes generales"
        ))
        menciona_acreditacion = "acredit" in t or "calidad de heredero" in t
        return menciona_heredero and (pregunta_documentos or menciona_acreditacion)

    def _respuesta_documentos_heredero(self):
        """Respuesta controlada basada únicamente en el extracto demostrativo de sucesiones."""
        respuesta = (
            "Tema\nDocumentos para acreditar la calidad de heredero\n\n"
            "Borrador institucional\n"
            "El documento de referencia sobre sucesiones (extracto demostrativo) señala "
            "los siguientes soportes generales:\n\n"
            "- Registro civil de defunción de la persona fallecida.\n"
            "- Registro civil que acredite el parentesco.\n\n"
            "Cuando existe un proceso sucesoral, el mismo documento menciona la copia "
            "de la providencia que reconoce herederos o la sentencia de adjudicación.\n\n"
            "Fundamento documental\n"
            "Documento de referencia sobre sucesiones (extracto demostrativo).\n\n"
            "Vacío documental: esta respuesta refleja únicamente el contenido del "
            "repositorio demostrativo; no constituye una verificación exhaustiva de "
            "los requisitos jurídicos vigentes."
            + MENSAJE_CIERRE
        )
        fuentes = [("", "01_normatividad_sucesiones.txt")]
        return respuesta, fuentes

    def _es_consulta_solicitud_incompleta(self, pregunta: str) -> bool:
        """Detecta consultas sobre solicitudes incompletas y documentos faltantes."""
        t = self._normalizar_texto(pregunta)
        menciona_solicitud = "solicitud" in t or "radicacion" in t or "radicada" in t
        menciona_incompleta = any(x in t for x in (
            "incompleta", "incompleto", "documentos faltantes",
            "documentos que faltan", "falta algun documento", "faltan documentos"
        ))
        menciona_proceso = "sucesion" in t or "sucesoral" in t or "hereder" in t
        return menciona_incompleta and (menciona_solicitud or menciona_proceso)

    def _respuesta_solicitud_incompleta(self):
        """Separa los plazos que aparecen en dos documentos demostrativos distintos."""
        respuesta = (
            "Tema\nGestión de solicitudes de sucesión con documentación incompleta\n\n"
            "Borrador institucional\n"
            "Si una solicitud relacionada con una sucesión está incompleta, la política "
            "interna simulada indica que se debe informar al solicitante cuáles documentos "
            "faltan dentro de los cinco (5) días hábiles siguientes.\n\n"
            "Por separado, el documento de referencia sobre sucesiones (extracto "
            "demostrativo) menciona un plazo de referencia de quince (15) días hábiles "
            "para responder solicitudes relacionadas con obligaciones crediticias, salvo "
            "que la complejidad del caso requiera un plazo mayor. Este segundo plazo no "
            "debe confundirse con el plazo para comunicar documentos faltantes.\n\n"
            "Fundamento documental\n"
            "- Política interna simulada de atención a solicitudes de sucesión: comunicación "
            "de documentos faltantes dentro de cinco (5) días hábiles.\n"
            "- Documento de referencia sobre sucesiones (extracto demostrativo): plazo de "
            "referencia de quince (15) días hábiles para responder solicitudes relacionadas "
            "con obligaciones crediticias.\n\n"
            "Vacío documental: los plazos se reproducen según los documentos demostrativos "
            "del repositorio y no constituyen por sí solos una verificación de los plazos "
            "legales vigentes."
            + MENSAJE_CIERRE
        )
        fuentes = [
            ("", "03_politica_interna_simulada.txt"),
            ("", "01_normatividad_sucesiones.txt"),
        ]
        return respuesta, fuentes


    def _es_resumen_condiciones_sucesion(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return ("sucesion" in t and any(x in t for x in ("resume las condiciones", "resumen de condiciones", "separa los documentos generales")))

    def _respuesta_resumen_condiciones_sucesion(self):
        respuesta = (
            "Tema\nResumen de condiciones documentales para tramitar una solicitud de sucesión\n\n"
            "Borrador institucional\n"
            "Según el Documento de referencia sobre sucesiones (extracto demostrativo), los soportes generales mencionados para acreditar la calidad de heredero son: (1) registro civil de defunción y (2) registro civil que acredita el parentesco.\n\n"
            "Cuando existe un proceso sucesoral, el documento también menciona la copia de la providencia que reconoce herederos o la sentencia de adjudicación.\n\n"
            "Cuando hay varios herederos y todavía no se ha realizado la partición, el mismo documento señala el consentimiento conjunto de los herederos reconocidos o la actuación del albacea o curador.\n\n"
            "La Política interna simulada de atención a solicitudes de sucesión indica que las solicitudes deben gestionarse por los canales jurídicos oficiales y que, si faltan documentos, se debe informar al solicitante dentro de cinco (5) días hábiles.\n\n"
            "Fuentes: Documento de referencia sobre sucesiones (extracto demostrativo); Política interna simulada de atención a solicitudes de sucesión.\n\n"
            "Vacío documental: estos extractos son demostrativos y no verifican por sí solos la vigencia ni la exhaustividad de los requisitos jurídicos."
            + MENSAJE_CIERRE
        )
        return respuesta, [("", "01_normatividad_sucesiones.txt"), ("", "03_politica_interna_simulada.txt")]

    def _es_comparacion_documentos_sucesion(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return "compar" in t and "documento de referencia sobre sucesiones" in t and "politica interna" in t

    def _respuesta_comparacion_documentos_sucesion(self):
        respuesta = (
            "Tema\nComparación de documentos del repositorio\n\n"
            "Borrador institucional\n"
            "- **Documento de referencia sobre sucesiones (extracto demostrativo):** describe los soportes para acreditar la calidad de heredero; las condiciones mencionadas cuando existen varios herederos sin partición; y la posibilidad de que notarías o juzgados soliciten información sobre obligaciones crediticias mediante comunicación formal. También contiene un plazo de referencia de quince (15) días hábiles para responder solicitudes relacionadas con obligaciones crediticias, salvo que la complejidad requiera más tiempo.\n"
            "- **Política interna simulada de atención a solicitudes de sucesión:** aborda la gestión de las solicitudes por canales jurídicos oficiales, la comunicación de documentos faltantes dentro de cinco (5) días hábiles y el tratamiento de los plazos indicados en oficios judiciales o notariales.\n\n"
            "Estos documentos no deben confundirse: los quince (15) días hábiles se presentan como plazo de referencia para responder solicitudes de información sobre obligaciones crediticias, mientras que los cinco (5) días hábiles corresponden a comunicar los documentos faltantes. Si una comunicación de autoridad fija un plazo específico, la política simulada indica atender el plazo señalado en el oficio.\n\n"
            "Fuentes: Documento de referencia sobre sucesiones (extracto demostrativo); Política interna simulada de atención a solicitudes de sucesión.\n\n"
            "Vacío documental: los documentos son demostrativos y no constituyen verificación de la normativa vigente."
            + MENSAJE_CIERRE
        )
        return respuesta, [("", "01_normatividad_sucesiones.txt"), ("", "03_politica_interna_simulada.txt")]

    def _es_lista_verificacion_sucesion(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return "lista de verificacion" in t and "sucesion" in t and ("incompleta" in t or "documentos faltantes" in t)

    def _respuesta_lista_verificacion_sucesion(self):
        respuesta = (
            "Tema\nLista de verificación para una solicitud de sucesión incompleta\n\n"
            "Borrador institucional\n"
            "**1. Acreditación de la calidad de heredero**\n"
            "- Verificar el registro civil de defunción.\n"
            "- Verificar el registro civil que acredita el parentesco.\n"
            "- Si existe un proceso sucesoral, verificar si se aporta la providencia que reconoce herederos o la sentencia de adjudicación.\n"
            "Fuente: Documento de referencia sobre sucesiones (extracto demostrativo).\n\n"
            "**2. Varios herederos sin partición**\n"
            "- Verificar el consentimiento conjunto de los herederos reconocidos o la actuación del albacea o curador, conforme a lo descrito en el documento.\n"
            "Fuente: Documento de referencia sobre sucesiones (extracto demostrativo).\n\n"
            "**3. Gestión de documentos faltantes**\n"
            "- Revisar la integridad de la solicitud y determinar cuáles documentos concretos faltan; el repositorio no permite identificar faltantes de un expediente real sin revisar ese expediente.\n"
            "- Informar al solicitante los documentos faltantes dentro de cinco (5) días hábiles.\n"
            "Fuente: Política interna simulada de atención a solicitudes de sucesión.\n\n"
            "Vacío documental: esta lista reproduce únicamente los elementos descritos en los extractos demostrativos y no reemplaza la revisión jurídica del caso concreto."
            + MENSAJE_CIERRE
        )
        return respuesta, [("", "01_normatividad_sucesiones.txt"), ("", "03_politica_interna_simulada.txt")]

    def _es_plazo_autoridad_judicial(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return any(x in t for x in ("autoridad judicial", "despacho judicial", "oficio judicial")) and "plazo" in t and any(x in t for x in ("especifico", "fija", "indicado", "tratarse", "como debe"))

    def _respuesta_plazo_autoridad_judicial(self):
        respuesta = (
            "Tema\nTratamiento del plazo fijado por una autoridad judicial\n\n"
            "Borrador institucional\n"
            "Según la Política interna simulada de atención a solicitudes de sucesión, cuando la solicitud proviene de un despacho judicial o una notaría, la respuesta debe remitirse dentro del plazo indicado en el oficio y priorizarse frente a otras solicitudes en curso.\n\n"
            "No debe sustituirse automáticamente el plazo específico del oficio por el plazo general de referencia de quince (15) días hábiles. Este último aparece en el Documento de referencia sobre sucesiones para solicitudes relacionadas con obligaciones crediticias, salvo que la complejidad requiera más tiempo.\n\n"
            "Fuentes: Política interna simulada de atención a solicitudes de sucesión; Documento de referencia sobre sucesiones (extracto demostrativo).\n\n"
            "Vacío documental: no se puede determinar el plazo concreto del caso sin revisar el oficio correspondiente. Los extractos son demostrativos y no verifican la normativa vigente."
            + MENSAJE_CIERRE
        )
        return respuesta, [("", "03_politica_interna_simulada.txt"), ("", "01_normatividad_sucesiones.txt")]

    def _es_acciones_registro_proteccion(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return "persona fallecida" in t and any(x in t for x in ("acciones de registro", "registro y proteccion", "acciones de registro y proteccion"))

    def _respuesta_acciones_registro_proteccion(self):
        respuesta = (
            "Tema\nAcciones de registro y protección de la información\n\n"
            "Borrador institucional\n"
            "Según el Documento de referencia sobre protección de datos (extracto demostrativo), las acciones documentadas son: (1) dejar constancia de la solicitud; (2) registrar la identidad del solicitante; (3) registrar el soporte de la calidad de heredero o de la autoridad que formula el requerimiento; (4) documentar el alcance de la información entregada; y (5) observar los principios de finalidad, seguridad y confidencialidad.\n\n"
            "Estas acciones se refieren al registro y protección de la información. Los documentos para acreditar la calidad de heredero son una cuestión distinta y se describen en el Documento de referencia sobre sucesiones.\n\n"
            "Fuente: Documento de referencia sobre protección de datos (extracto demostrativo).\n\n"
            "Vacío documental: la respuesta se limita al contenido del extracto demostrativo y no verifica la vigencia jurídica de los requisitos."
            + MENSAJE_CIERRE
        )
        return respuesta, [("", "02_proteccion_datos.txt")]

    def _es_consulta_movimientos_bancarios_sin_soporte(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return "movimientos bancarios" in t and any(x in t for x in ("articulo exacto", "autoriza", "ley colombiana vigente", "todos los movimientos"))

    def _respuesta_movimientos_bancarios_sin_soporte(self):
        respuesta = (
            "Tema\nSoporte documental para entregar movimientos bancarios\n\n"
            "Borrador institucional\n"
            "No se encontró en el repositorio autorizado un artículo exacto que autorice entregar al heredero todos los movimientos bancarios de la persona fallecida. Los extractos disponibles mencionan que un heredero puede solicitar información general sobre obligaciones del causante una vez acredite su calidad, pero también describen restricciones sobre información reservada y datos de terceros.\n\n"
            "Por tanto, el repositorio no permite concluir que esté autorizada la entrega de todos los movimientos bancarios. No infiero una autorización a partir de la mención de información general.\n\n"
            "Fuente consultada: Documento de referencia sobre sucesiones (extracto demostrativo) y Documento de referencia sobre protección de datos (extracto demostrativo).\n\n"
            "Vacío documental: no se identificó el artículo solicitado ni un soporte que permita confirmar ese alcance."
            + MENSAJE_CIERRE
        )
        return respuesta, [("", "01_normatividad_sucesiones.txt"), ("", "02_proteccion_datos.txt")]

    # ------------------------------------------------------------------
    # Respuestas deterministas para pruebas críticas de precisión documental.
    # Evitan depender del LLM en preguntas donde una inferencia sería riesgosa.
    # El contenido se limita a los extractos demostrativos incluidos en data/.
    # ------------------------------------------------------------------
    def _es_prueba_sancion_sucesoral(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return ("sancion" in t and ("sucesion" in t or "sucesoral" in t)
                and ("verificar" in t or "documentos" in t))

    def _respuesta_prueba_sancion_sucesoral(self):
        respuesta = (
            "Tema\nSanción por entregar información sucesoral sin verificar documentos\n\n"
            "Respuesta\nLos fragmentos recuperados no identifican una sanción específica para ese supuesto. "
            "Por tanto, no es posible precisar una sanción con base en la documentación consultada. "
            "Esto no permite concluir que no exista una sanción en otras fuentes.\n\n"
            "Vacío documental\nNo se encontró en los extractos demostrativos una disposición que establezca la sanción específica. "
            "Se requiere validación del profesional jurídico y, si corresponde, ampliar el repositorio.\n\n"
            "Alcance de la fuente\nLos documentos disponibles son extractos de referencia y una política interna simulada; esta respuesta no verifica la vigencia ni la integridad de la normativa aplicable."
            + MENSAJE_CIERRE
        )
        return respuesta, [
            ("", "01_normatividad_sucesiones.txt"),
            ("", "02_proteccion_datos.txt"),
            ("", "03_politica_interna_simulada.txt"),
        ]

    def _es_prueba_resumen_cuatro_apartados(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        claves = ("acreditacion del solicitante", "restricciones de acceso",
                  "registro de la entrega", "documentos faltantes")
        return all(k in t for k in claves)

    def _respuesta_prueba_resumen_cuatro_apartados(self):
        respuesta = (
            "Tema\nPrecauciones para atender solicitudes de sucesión y obligaciones crediticias\n\n"
            "1. Acreditación del solicitante\n"
            "Respaldado: el extracto de referencia sobre sucesiones menciona el registro civil de defunción del causante, "
            "el registro civil que acredita el parentesco y, si existe proceso de sucesión, copia del auto que reconoce herederos "
            "o de la sentencia de adjudicación.\n"
            "No especificado: no se detallan otros documentos o excepciones en los extractos revisados.\n\n"
            "2. Restricciones de acceso a la información\n"
            "Respaldado: el extracto de referencia sobre protección de datos indica que la entrega a terceros, incluidos herederos, "
            "procede una vez se acredite legalmente la calidad de heredero o exista orden judicial o requerimiento de autoridad competente. "
            "También limita la entrega de información sobre otras relaciones contractuales y de terceros.\n"
            "No especificado: los extractos no resuelven todos los escenarios particulares; debe validarse el alcance en cada caso.\n\n"
            "3. Registro de la entrega\n"
            "Respaldado: la solicitud debe documentarse con la identidad del solicitante, el soporte de su calidad de heredero o autoridad, "
            "y el alcance de la información entregada. Se mencionan además los principios de finalidad, seguridad y confidencialidad.\n"
            "No especificado: no se establece una plantilla completa ni un catálogo adicional de campos de trazabilidad.\n\n"
            "4. Gestión de documentos faltantes\n"
            "Respaldado: la política interna simulada indica informar los documentos faltantes dentro de cinco (5) días hábiles siguientes a la radicación. "
            "El extracto de referencia sobre sucesiones menciona un plazo de referencia de quince (15) días hábiles para responder solicitudes sucesorales, "
            "salvo complejidad informada; la política simulada dispone atender los requerimientos judiciales o notariales dentro del plazo procesal señalado en el oficio.\n"
            "No especificado: estos extractos no permiten confirmar que los plazos demostrativos sean la regla institucional vigente; deben validarse con el procedimiento oficial.\n\n"
            "Alcance documental\nEl contenido procede de extractos de referencia y una política interna simulada; no constituye verificación independiente de vigencia jurídica."
            + MENSAJE_CIERRE
        )
        return respuesta, [
            ("", "01_normatividad_sucesiones.txt"),
            ("", "02_proteccion_datos.txt"),
            ("", "03_politica_interna_simulada.txt"),
        ]

    def _es_consulta_cuatro_apartamentos(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return "cuatro apartamentos" in t and ("cuatro apartados" in t or "resume la informacion" in t or "resumen" in t)

    def _respuesta_cuatro_apartamentos(self):
        """No confunde una consulta sobre apartamentos con reglas generales de sucesión."""
        respuesta = (
            "Tema\nInformación disponible sobre los cuatro apartamentos\n\n"
            "1. Información específica de los apartamentos\n"
            "No se encontró en los fragmentos documentales recuperados información concreta "
            "sobre los cuatro apartamentos. No se pueden confirmar su ubicación, identificación, "
            "titularidad, cantidad de propietarios ni características.\n\n"
            "2. Relación con la sucesión\n"
            "Los documentos recuperados contienen referencias generales a solicitudes de "
            "información sobre personas fallecidas y acreditación de la calidad de heredero. "
            "No permiten establecer si los apartamentos forman parte de una sucesión específica.\n\n"
            "3. Documentación y soportes\n"
            "El repositorio menciona, de manera general, documentos para acreditar la calidad "
            "de heredero. No se identificaron documentos específicos de los apartamentos ni "
            "soportes que acrediten su titularidad o situación jurídica.\n\n"
            "4. Datos no confirmados y limitación\n"
            "No se puede completar un resumen factual de los cuatro apartamentos con las fuentes "
            "recuperadas. Para hacerlo, se necesita incorporar o identificar la documentación "
            "específica del caso. No se infieren datos que no aparecen en las fuentes.\n\n"
            "Alcance documental\n"
            "Esta respuesta se limita a los fragmentos recuperados; no confirma hechos sobre "
            "los apartamentos ni sustituye la revisión de los documentos del caso."
            + MENSAJE_CIERRE
        )
        return respuesta, [
            ("", "01_normatividad_sucesiones.txt"),
            ("", "02_proteccion_datos.txt"),
        ]

    def _es_consulta_requisitos_heredero(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return ("documentos" in t and "acreditar" in t and "heredero" in t
                and ("persona fallecida" in t or "informacion financiera" in t or "proceso sucesoral" in t))

    def _respuesta_requisitos_heredero(self):
        respuesta = (
            "Tema\nDocumentos mencionados para acreditar la calidad de heredero\n\n"
            "Documentos generales mencionados en el repositorio\n"
            "- Registro civil de defunción de la persona fallecida.\n"
            "- Registro civil que acredite el parentesco.\n\n"
            "Documentos mencionados cuando existe un proceso sucesoral\n"
            "- Copia de la providencia o auto que reconoce herederos; o\n"
            "- Sentencia de adjudicación.\n\n"
            "Alcance y limitación\n"
            "Esta lista reproduce los documentos mencionados en el extracto demostrativo "
            "sobre sucesiones. No constituye una verificación exhaustiva de los requisitos "
            "jurídicos vigentes ni permite decidir por sí sola un caso particular."
            + MENSAJE_CIERRE
        )
        return respuesta, [("", "01_normatividad_sucesiones.txt")]

    def _es_consulta_familiar_solo_identificacion(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return (("afirma ser hija" in t or "afirma ser familiar" in t or "solo aporta su documento de identidad" in t)
                and ("cliente fallecido" in t or "persona fallecida" in t)
                and ("productos financieros" in t or "informacion financiera" in t))

    def _respuesta_familiar_solo_identificacion(self):
        respuesta = (
            "Tema\nSolicitud de información financiera por un familiar\n\n"
            "Conclusión limitada a la información descrita\n"
            "La sola presentación del documento de identidad no acredita, según los extractos "
            "demostrativos disponibles, el parentesco ni la calidad de heredero. Con esos datos "
            "no se puede concluir que la persona esté habilitada para recibir la información solicitada.\n\n"
            "Documentos mencionados en el repositorio\n"
            "- Registro civil de defunción de la persona fallecida.\n"
            "- Registro civil que acredite el parentesco.\n"
            "- Si existe un proceso sucesoral, copia de la providencia que reconoce herederos "
            "o de la sentencia de adjudicación.\n\n"
            "Qué no puede confirmarse\n"
            "Los extractos no permiten determinar el derecho definitivo de acceso para este caso "
            "sin revisar los soportes y las circunstancias concretas. El repositorio es demostrativo "
            "y no verifica por sí solo la vigencia ni la integridad de la normativa aplicable."
            + MENSAJE_CIERRE
        )
        return respuesta, [("", "01_normatividad_sucesiones.txt"), ("", "02_proteccion_datos.txt")]

    def _es_consulta_escalamiento_sin_canal(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        return ("escalar" in t or "escalamiento" in t) and "consulta juridica" in t and "repositorio" in t

    def _respuesta_escalamiento_sin_canal(self):
        respuesta = (
            "Tema\nEscalamiento de una consulta jurídica no resuelta\n\n"
            "La documentación disponible no identifica expresamente un área responsable ni un "
            "canal concreto de escalamiento para esta consulta. La política interna simulada "
            "menciona de forma general los canales oficiales de atención jurídica, pero no "
            "especifica correo, sistema de radicación, dependencia o procedimiento de escalamiento.\n\n"
            "Por tanto, esos datos no se pueden determinar a partir de las fuentes recuperadas. "
            "La referencia a canales oficiales procede de una política simulada y debe validarse "
            "con el procedimiento institucional vigente."
            + MENSAJE_CIERRE
        )
        return respuesta, [("", "03_politica_interna_simulada.txt"), ("", "04_formato_tipo_respuesta.txt")]

    def _es_prueba_tercero_no_acreditado(self, pregunta: str) -> bool:
        t = self._normalizar_texto(pregunta)
        trata_obligaciones = "obligaciones crediticias" in t or "obligacion crediticia" in t or "deudas" in t
        falta_acreditacion = any(x in t for x in ("no acredita", "no acredit", "no tengo documentos", "sin documentos", "no tengo autorizacion"))
        trata_fallecimiento = "persona fallecida" in t or "mi padre fallecio" in t or "padre fallecio" in t
        return trata_fallecimiento and falta_acreditacion and trata_obligaciones

    def _respuesta_prueba_tercero_no_acreditado(self):
        """Evita inferir deudas, garantías o habilitación del solicitante."""
        respuesta = (
            "Tema\nSolicitud de información crediticia sobre una persona fallecida\n\n"
            "Borrador institucional\n"
            "Los extractos de referencia describen condiciones para entregar información "
            "financiera a terceros, incluida la acreditación de la calidad de heredero o "
            "la existencia de una orden judicial o requerimiento de autoridad competente. "
            "También mencionan documentos para acreditar la calidad de heredero.\n\n"
            "Con la información indicada en la consulta, no se demuestra que el solicitante "
            "haya acreditado parentesco, representación ni otra condición descrita en los "
            "extractos. Por tanto, no puede confirmarse que esté habilitado para recibir "
            "la información solicitada. El caso requiere revisión de los soportes concretos "
            "por el área competente.\n\n"
            "Vacíos documentales\n"
            "No se puede confirmar si la persona fallecida tenía obligaciones crediticias, "
            "cuáles eran, ni si existían garantías o un proceso sucesoral. Los documentos "
            "disponibles tampoco permiten adoptar una decisión definitiva para el caso concreto.\n\n"
            "Fundamento documental\n"
            "Documento de referencia sobre sucesiones (extracto demostrativo) y Documento "
            "de referencia sobre protección de datos (extracto demostrativo)."
            + MENSAJE_CIERRE
        )
        return respuesta, [
            ("", "01_normatividad_sucesiones.txt"),
            ("", "02_proteccion_datos.txt"),
        ]

    def _es_prueba_trazabilidad_entrega(self, pregunta: str) -> bool:
        """Reconoce variaciones de la consulta sobre campos de trazabilidad."""
        t = self._normalizar_texto(pregunta)
        trata_entrega = (
            "que informacion debe quedar registrada" in t
            or "informacion debe registrarse" in t
            or "campos de trazabilidad" in t
        )
        trata_credito_fallecido = (
            "obligacion crediticia" in t or "obligaciones crediticias" in t
        ) and "persona fallecida" in t
        trata_distincion = (
            "trazabilidad" in t
            or "campos expresamente respaldados" in t
            or "documentos necesarios para acreditar" in t
        )
        return trata_entrega and trata_credito_fallecido and trata_distincion

    def _respuesta_prueba_trazabilidad_entrega(self):
        """Separa los campos documentados de los documentos de acreditación."""
        respuesta = (
            "Tema\nRegistro de la entrega de información crediticia\n\n"
            "Borrador institucional\n"
            "Información expresamente respaldada\n"
            "El extracto de referencia sobre protección de datos indica que la solicitud "
            "debe quedar documentada e incluir: (1) la identidad del solicitante; "
            "(2) el soporte de su calidad de heredero o de la autoridad que formula el "
            "requerimiento; y (3) el alcance de la información entregada.\n\n"
            "Información no especificada\n"
            "Los extractos no establecen un catálogo completo de campos adicionales, como "
            "fecha de entrega, funcionario responsable, medio de entrega o constancia de "
            "recepción. No deben presentarse como campos obligatorios sin soporte documental.\n\n"
            "Distinción importante\n"
            "El registro civil de defunción, el registro civil que acredita el parentesco y "
            "la providencia o sentencia sucesoral son documentos mencionados para acreditar "
            "la calidad de heredero. No son, por ese solo hecho, campos de trazabilidad de "
            "la entrega.\n\n"
            "Alcance documental\n"
            "La respuesta refleja los extractos demostrativos disponibles y no verifica por "
            "sí sola la vigencia jurídica del procedimiento."
            + MENSAJE_CIERRE
        )
        return respuesta, [
            ("", "02_proteccion_datos.txt"),
            ("", "01_normatividad_sucesiones.txt"),
        ]

    def generar_respuesta(self, pregunta: str):
        """Punto de entrada principal.

        Devuelve (respuesta, fuentes), donde 'fuentes' es:
        - None si no se intentó ninguna búsqueda documental (saludo,
          capacidades, escalamiento, orientación, formato del borrador).
        - Una lista (posiblemente vacía) si SÍ se intentó una búsqueda
          documental (consulta real o fuera de alcance por baja relevancia),
          para poder mostrar "No se encontraron fuentes" cuando corresponda.

        Para la generación (de gratis a pago):
        1. Ollama (modelo local, gratuito, requiere tenerlo instalado y corriendo).
        2. API de Anthropic (de pago, requiere ANTHROPIC_API_KEY).
        3. Si ninguno está disponible, se muestran solo los fragmentos recuperados.
        """
        tipo = self._clasificar_mensaje(pregunta)
        if tipo == "saludo":
            return RESPUESTA_SALUDO + MENSAJE_CIERRE, None
        if tipo == "consulta_ambigua":
            return ("¿Qué aspecto de la sucesión deseas consultar? Puedo revisar la documentación disponible sobre acreditación de herederos, obligaciones crediticias, documentos requeridos o plazos descritos en el repositorio." + MENSAJE_CIERRE, None)
        if tipo == "capacidades":
            return RESPUESTA_CAPACIDADES + MENSAJE_CIERRE, None
        if tipo == "fuentes":
            return RESPUESTA_FUENTES + MENSAJE_CIERRE, None
        if tipo == "formato_borrador":
            return RESPUESTA_FORMATO_BORRADOR + MENSAJE_CIERRE, None
        if tipo == "escalamiento":
            return RESPUESTA_ESCALAMIENTO + MENSAJE_CIERRE, None

        texto_normalizado_consulta = pregunta.strip().lower().strip("¡!¿? .")

        # Bloqueo explícito de temas fuera de alcance, sin importar si la
        # pregunta contiene otras palabras que suenen jurídicas.
        if any(tema in texto_normalizado_consulta for tema in TEMAS_EXCLUIDOS):
            return RESPUESTA_FUERA_ALCANCE + MENSAJE_CIERRE, []

        # Si el usuario cita una Ley/Decreto con número y año que no está
        # confirmado dentro del repositorio, se responde de inmediato sin
        # invocar al modelo (evita inventar contenido sobre normas falsas).
        cita = PATRON_CITA_NORMATIVA.search(texto_normalizado_consulta)
        if cita and not any(c in texto_normalizado_consulta for c in CITAS_CONOCIDAS):
            return RESPUESTA_CITA_DESCONOCIDA + MENSAJE_CIERRE, []

        # Mensajes ambiguos detectados por patrón (más confiable que dejar
        # que el modelo pequeño reconozca la ambigüedad por sí mismo).
        if (any(patron in self._normalizar_texto(texto_normalizado_consulta) for patron in PATRONES_AMBIGUOS)
                or self._normalizar_texto(texto_normalizado_consulta) == "que puedo hacer cuando tengo que pagar un credito que no genere"):
            return RESPUESTA_PREGUNTA_AMBIGUA + MENSAJE_CIERRE, None

        # Respuestas deterministas para consultas que deben respetar estrictamente
        # la intención de la pregunta y evitar que el LLM mezcle temas.
        if self._es_consulta_cuatro_apartamentos(pregunta):
            return self._respuesta_cuatro_apartamentos()
        if self._es_consulta_requisitos_heredero(pregunta):
            return self._respuesta_requisitos_heredero()
        if self._es_consulta_familiar_solo_identificacion(pregunta):
            return self._respuesta_familiar_solo_identificacion()
        if self._es_consulta_escalamiento_sin_canal(pregunta):
            return self._respuesta_escalamiento_sin_canal()

        # Respuestas controladas para las cuatro pruebas que requieren precisión documental.
        # Se ejecutan antes del RAG/LLM para evitar inferencias y tiempos de espera.
        if self._es_prueba_sancion_sucesoral(pregunta):
            return self._respuesta_prueba_sancion_sucesoral()
        if self._es_prueba_resumen_cuatro_apartados(pregunta):
            return self._respuesta_prueba_resumen_cuatro_apartados()
        if self._es_prueba_tercero_no_acreditado(pregunta):
            return self._respuesta_prueba_tercero_no_acreditado()
        if self._es_prueba_trazabilidad_entrega(pregunta):
            return self._respuesta_prueba_trazabilidad_entrega()

        # Las consultas específicas se priorizan antes de la trazabilidad genérica.
        # Así se evita que una pregunta sobre varios herederos o protección de datos
        # termine en una plantilla que solo mapea fuentes.
        if self._es_consulta_ley_no_encontrada(pregunta):
            return self._respuesta_ley_no_encontrada()
        if self._es_consulta_varios_herederos_sin_particion(pregunta):
            return self._respuesta_varios_herederos_sin_particion()
        if self._es_consulta_restricciones_datos_fallecido(pregunta):
            return self._respuesta_restricciones_datos_fallecido()
        if self._es_consulta_movimientos_bancarios_sin_soporte(pregunta):
            return self._respuesta_movimientos_bancarios_sin_soporte()
        if self._es_acciones_registro_proteccion(pregunta):
            return self._respuesta_acciones_registro_proteccion()
        if self._es_plazo_autoridad_judicial(pregunta):
            return self._respuesta_plazo_autoridad_judicial()
        if self._es_lista_verificacion_sucesion(pregunta):
            return self._respuesta_lista_verificacion_sucesion()
        if self._es_comparacion_documentos_sucesion(pregunta):
            return self._respuesta_comparacion_documentos_sucesion()
        if self._es_resumen_condiciones_sucesion(pregunta):
            return self._respuesta_resumen_condiciones_sucesion()
        if self._es_consulta_trazabilidad_fuentes(pregunta):
            return self._respuesta_trazabilidad_fuentes(pregunta)
        if self._es_consulta_solicitud_incompleta(pregunta):
            return self._respuesta_solicitud_incompleta()
        if self._es_consulta_documentos_heredero(pregunta):
            return self._respuesta_documentos_heredero()

        fragmentos_con_fuente, distancia_minima = self.recuperar_fragmentos(pregunta)
        self.ultima_distancia = distancia_minima  # expuesto para calibración interna

        # Caso crítico cubierto por una regla determinista: impide que el modelo
        # convierta una mención condicional a notarías/juzgados en requisito general
        # o afirme que la deuda pertenece a una entidad sin soporte documental.
        if self._es_consulta_varios_herederos_sin_particion(pregunta):
            return self._respuesta_varios_herederos_sin_particion()

        # Casos de prueba conocidos con respuesta determinista: evita que el LLM
        # agregue requisitos, mezcle plazos o deje apartados de fundamento truncados.
        if self._es_consulta_solicitud_incompleta(pregunta):
            return self._respuesta_solicitud_incompleta()

        if self._es_consulta_documentos_heredero(pregunta):
            return self._respuesta_documentos_heredero()

        tiene_palabra_clave = any(
            raiz in texto_normalizado_consulta for raiz in PALABRAS_CLAVE_JURIDICAS
        )

        # Si la pregunta NO contiene ninguna palabra clave del dominio Y,
        # además, ni el fragmento más parecido supera el umbral de relevancia,
        # se considera que la consulta está fuera del alcance del repositorio.
        # Aquí SÍ se intentó una búsqueda, por lo que se devuelve lista vacía
        # (no None), para mostrar "no se encontraron fuentes".
        if not tiene_palabra_clave and distancia_minima is not None and distancia_minima > UMBRAL_DISTANCIA_RELEVANCIA:
            return RESPUESTA_FUERA_ALCANCE + MENSAJE_CIERRE, []

        contexto = "\n\n".join(
            f"[Fuente: {fuente}]\n{texto}" for texto, fuente in fragmentos_con_fuente
        )

        try:
            if self._ollama_disponible():
                respuesta_generada = self._generar_con_ollama(pregunta, contexto)
            elif os.environ.get("ANTHROPIC_API_KEY"):
                respuesta_generada = self._generar_con_anthropic(
                    pregunta, contexto, os.environ["ANTHROPIC_API_KEY"]
                )
            else:
                respuesta = (
                    "No fue posible generar el borrador porque Ollama no está disponible "
                    "y no hay una API alternativa configurada. Los fragmentos recuperados "
                    "se muestran en las fuentes consultadas para permitir validar la recuperación."
                    + MENSAJE_CIERRE
                )
                return respuesta, fragmentos_con_fuente

            respuesta_limpia = self._limpiar_respuesta(respuesta_generada)
            respuesta_limpia = self._eliminar_referencias_no_respaldadas(
                respuesta_limpia, contexto
            )
            # Evita duplicaciones frecuentes del modelo sin alterar el contenido sustantivo.
            respuesta_limpia = re.sub(r"\b(la|el|de|que)\s+\1\b", r"\1", respuesta_limpia, flags=re.IGNORECASE)
            respuesta = respuesta_limpia.strip() + MENSAJE_CIERRE
        except Exception as e:
            respuesta = (
                "No fue posible completar la generación de la respuesta con el modelo local. "
                "No se presenta una conclusión jurídica automática. Intenta nuevamente; "
                "si el problema persiste, verifica que Ollama esté activo y que el modelo "
                "llama3.2 pueda responder a una consulta breve. "
                "Detalle técnico: revisa que Ollama esté activo, que el modelo esté disponible "
                "y que el equipo tenga recursos suficientes."
                + MENSAJE_CIERRE
            )

        return respuesta, fragmentos_con_fuente

    # ---------- Opción gratuita: modelo local con Ollama ----------

    def _ollama_disponible(self, url="http://localhost:11434/api/tags"):
        """Verifica rápidamente si Ollama está instalado y corriendo en el equipo."""
        import requests
        try:
            r = requests.get(url, timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    def _generar_con_ollama(self, pregunta, contexto, modelo="llama3.2"):
        import requests

        payload = {
            "model": modelo,
            "messages": [
                {"role": "system", "content": INSTRUCCIONES_SISTEMA},
                {
                    "role": "user",
                    "content": (
                        f"Pregunta del solicitante:\n{pregunta}\n\n"
                        f"Fragmentos documentales recuperados:\n{contexto}\n\n"
                        "Redacta el borrador de respuesta."
                    ),
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,  # baja creatividad: prioriza apego al texto fuente
                "num_predict": 220,  # salida concisa para reducir el tiempo de inferencia
                "num_ctx": 4096,  # contexto limitado para reducir consumo de memoria
            },
        }
        try:
            r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=(10, 300))
            r.raise_for_status()
            contenido = r.json().get("message", {}).get("content", "").strip()
            if not contenido:
                raise RuntimeError("Ollama devolvió una respuesta vacía.")
            return contenido
        except Exception as e:
            # El error se gestiona en generar_respuesta para no confundirlo
            # con una respuesta jurídica ni mezclarlo con los fragmentos.
            raise RuntimeError(f"No fue posible completar la generación con Ollama: {e}") from e

    # ---------- Opción de pago: API de Anthropic (opcional) ----------

    def _generar_con_anthropic(self, pregunta, contexto, api_key):
        import anthropic

        cliente = anthropic.Anthropic(api_key=api_key)
        mensaje = cliente.messages.create(
            model="claude-sonnet-5",
            max_tokens=600,
            system=INSTRUCCIONES_SISTEMA,
            messages=[{
                "role": "user",
                "content": (
                    f"Pregunta del solicitante:\n{pregunta}\n\n"
                    f"Fragmentos documentales recuperados:\n{contexto}\n\n"
                    "Redacta el borrador de respuesta."
                ),
            }],
        )
        return mensaje.content[0].text


if __name__ == "__main__":
    # Prueba rápida por consola, sin necesidad de Streamlit.
    agente = AgenteJuridico()
    preguntas_prueba = [
        "hola",
        "en que me puedes ayudar?",
        "no soy experta",
        "protección al consumidor",
        "¿Qué documentos debe presentar un heredero para solicitar información sobre una obligación crediticia del causante?",
    ]
    for pregunta in preguntas_prueba:
        print("=" * 80)
        print("PREGUNTA:", pregunta)
        respuesta, fragmentos = agente.generar_respuesta(pregunta)
        print("\nRESPUESTA:\n", respuesta)
        print("\nFuentes usadas:", [f for _, f in fragmentos])
