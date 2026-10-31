"""Pruebas de regresión de reglas deterministas de JoLex IA.
Estas pruebas no requieren iniciar ChromaDB, descargar embeddings ni ejecutar Ollama.
"""
import sys
import types
import unittest

# Dependencias ligeras de sustitución: los tests no inicializan el repositorio RAG.
try:
    import chromadb  # noqa: F401
except ModuleNotFoundError:
    sys.modules["chromadb"] = types.ModuleType("chromadb")

try:
    import sentence_transformers  # noqa: F401
except ModuleNotFoundError:
    st = types.ModuleType("sentence_transformers")
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
    st.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = st

from agente import AgenteJuridico, TEMAS_EXCLUIDOS

class TestReglasDeterministas(unittest.TestCase):
    def setUp(self):
        self.agente = AgenteJuridico.__new__(AgenteJuridico)

    def test_documentos_heredero(self):
        q = "¿Qué documentos señala el repositorio autorizado para acreditar la calidad de heredero de una persona fallecida?"
        self.assertTrue(self.agente._es_consulta_documentos_heredero(q))
        respuesta, fuentes = self.agente._respuesta_documentos_heredero()
        self.assertIn("registro civil de defunción", respuesta.lower())
        self.assertIn("registro civil que acredite el parentesco", respuesta.lower())
        self.assertIn("sentencia de adjudicación", respuesta.lower())
        self.assertNotIn("Fundamento normativo", respuesta)
        self.assertEqual([f for _, f in fuentes], ["01_normatividad_sucesiones.txt"])

    def test_solicitud_incompleta(self):
        q = "Según la política interna disponible, ¿qué debe hacerse si una solicitud relacionada con una sucesión llega incompleta?"
        self.assertTrue(self.agente._es_consulta_solicitud_incompleta(q))
        respuesta, fuentes = self.agente._respuesta_solicitud_incompleta()
        self.assertIn("cinco (5) días hábiles", respuesta)
        self.assertIn("quince (15) días hábiles", respuesta)
        self.assertIn("no debe confundirse", respuesta)
        self.assertIn("Política interna simulada", respuesta)
        self.assertEqual([f for _, f in fuentes], [
            "03_politica_interna_simulada.txt", "01_normatividad_sucesiones.txt"
        ])

    def test_consulta_fuente_consentimiento_no_devuelve_plantilla_documentos(self):
        q = "¿De qué documento obtuviste la afirmación de que se requiere consentimiento conjunto cuando existen varios herederos y no se ha realizado la partición? Identifica el documento y explica qué parte de la respuesta respalda."
        self.assertTrue(self.agente._es_consulta_trazabilidad_fuentes(q))
        respuesta, fuentes = self.agente._respuesta_trazabilidad_fuentes(q)
        self.assertIn("consentimiento conjunto", respuesta.lower())
        self.assertIn("Documento de referencia sobre sucesiones", respuesta)
        self.assertNotIn("Registro civil de defunción", respuesta)
        self.assertEqual([f for _, f in fuentes], ["01_normatividad_sucesiones.txt"])

    def test_trazabilidad_notarias_juzgados_identifica_fuente_y_no_universaliza(self):
        q = ("¿Qué documento respalda la afirmación de que las notarías o los juzgados "
             "pueden solicitar información sobre obligaciones crediticias mediante "
             "comunicación formal? Identifica la fuente y señala si el repositorio "
             "establece que ese canal es obligatorio para todas las solicitudes.")
        self.assertTrue(self.agente._es_consulta_trazabilidad_fuentes(q))
        respuesta, fuentes = self.agente._respuesta_trazabilidad_fuentes(q)
        self.assertIn("notarías o juzgados", respuesta.lower())
        self.assertIn("comunicación formal", respuesta.lower())
        self.assertIn("no establece que ese canal sea obligatorio para todas las solicitudes", respuesta.lower())
        self.assertNotIn("No puedo identificar con suficiente precisión", respuesta)
        self.assertEqual([f for _, f in fuentes], ["01_normatividad_sucesiones.txt"])

    def test_consulta_multiafirmacion_responde_los_tres_puntos(self):
        q = "Relaciona cada afirmación con el documento del repositorio que la respalda: (1) acreditar el parentesco, (2) obtener consentimiento conjunto cuando hay varios herederos sin partición y (3) comunicar los documentos faltantes de una solicitud incompleta."
        self.assertTrue(self.agente._es_consulta_trazabilidad_fuentes(q))
        respuesta, fuentes = self.agente._respuesta_trazabilidad_fuentes(q)
        self.assertIn("Acreditar el parentesco", respuesta)
        self.assertIn("Consentimiento conjunto", respuesta)
        self.assertIn("Comunicar los documentos faltantes", respuesta)
        self.assertEqual(set(f for _, f in fuentes), {"01_normatividad_sucesiones.txt", "03_politica_interna_simulada.txt"})

    def test_sancion_penal_fuera_de_alcance(self):
        q = "¿Qué sanción penal se aplica a un delito no relacionado con sucesiones?"
        t = self.agente._normalizar_texto(q)
        self.assertTrue(any(tema in t for tema in TEMAS_EXCLUIDOS))

    def test_ley_01_sucesiones_no_se_inventa(self):
        q = "¿El repositorio contiene una norma identificada como Ley 01 de Sucesiones?"
        self.assertTrue(self.agente._es_consulta_ley_no_encontrada(q))
        respuesta, fuentes = self.agente._respuesta_ley_no_encontrada()
        self.assertIn("No se encontró", respuesta)
        self.assertEqual(fuentes, [])


    def test_varios_herederos_no_se_desvia_a_trazabilidad_generica(self):
        q = ("Según exclusivamente el repositorio autorizado, cuando existen varios herederos "
             "y todavía no se ha realizado la partición de bienes, ¿qué condiciones deben "
             "verificarse antes de entregar información sobre las obligaciones crediticias "
             "del causante? Identifica el documento que sustenta cada condición.")
        self.agente._clasificar_mensaje = lambda pregunta: "consulta"
        self.agente.recuperar_fragmentos = lambda pregunta: (_ for _ in ()).throw(
            AssertionError("La consulta específica debe resolverse antes de la recuperación RAG")
        )
        respuesta, fuentes = self.agente.generar_respuesta(q)
        self.assertIn("Acreditación de la calidad de heredero", respuesta)
        self.assertIn("Consentimiento en caso de varios herederos", respuesta)
        self.assertIn("Canal de trámite", respuesta)
        self.assertEqual({f for _, f in fuentes}, {"01_normatividad_sucesiones.txt", "03_politica_interna_simulada.txt"})

    def test_restricciones_datos_fallecido_responde_con_fuente_correcta(self):
        q = ("Según exclusivamente los documentos del repositorio, ¿qué restricciones se describen "
             "para entregar información de una persona fallecida a sus herederos? Identifica el "
             "documento de soporte y señala qué aspectos no pueden confirmarse con la información disponible.")
        self.assertTrue(self.agente._es_consulta_restricciones_datos_fallecido(q))
        respuesta, fuentes = self.agente._respuesta_restricciones_datos_fallecido()
        self.assertIn("información financiera", respuesta)
        self.assertIn("otras relaciones contractuales", respuesta)
        self.assertIn("Vacíos documentales", respuesta)
        self.assertEqual([f for _, f in fuentes], ["02_proteccion_datos.txt"])

    def test_pregunta_proteccion_datos_no_se_clasifica_como_trazabilidad_generica(self):
        q = ("Según exclusivamente los documentos del repositorio, ¿qué restricciones se describen "
             "para entregar información de una persona fallecida a sus herederos? Identifica el documento de soporte.")
        self.assertFalse(self.agente._es_consulta_trazabilidad_fuentes(q))
        self.assertTrue(self.agente._es_consulta_restricciones_datos_fallecido(q))

    def test_varios_herederos_prioriza_regla_especifica(self):
        q = "Cuando existen varios herederos y no se ha realizado la partición de bienes, ¿qué documentos se necesitan para entregar información sobre obligaciones crediticias?"
        self.assertTrue(self.agente._es_consulta_varios_herederos_sin_particion(q))
        self.assertFalse(self.agente._es_consulta_documentos_heredero(q))

    def test_resumen_condiciones_sucesion_responde_tres_escenarios(self):
        q = "Resume las condiciones documentales para tramitar una solicitud relacionada con una sucesión. Separa los documentos generales, los requisitos cuando existe un proceso sucesoral y las condiciones aplicables si hay varios herederos sin partición. Indica la fuente de cada punto."
        self.assertTrue(self.agente._es_resumen_condiciones_sucesion(q))
        respuesta, fuentes = self.agente._respuesta_resumen_condiciones_sucesion()
        for esperado in ("registro civil de defunción", "registro civil que acredita el parentesco", "sentencia de adjudicación", "consentimiento conjunto", "cinco (5) días hábiles"):
            self.assertIn(esperado, respuesta.lower())
        self.assertEqual({f for _, f in fuentes}, {"01_normatividad_sucesiones.txt", "03_politica_interna_simulada.txt"})

    def test_comparacion_documentos_distingue_plazos(self):
        q = "Compara el Documento de referencia sobre sucesiones con la Política interna simulada de atención a solicitudes de sucesión. ¿Qué aporta cada documento y qué aspectos no deben confundirse?"
        self.assertTrue(self.agente._es_comparacion_documentos_sucesion(q))
        respuesta, fuentes = self.agente._respuesta_comparacion_documentos_sucesion()
        self.assertIn("cinco (5) días hábiles", respuesta)
        self.assertIn("quince (15) días hábiles", respuesta)
        self.assertIn("no deben confundirse", respuesta)
        self.assertEqual({f for _, f in fuentes}, {"01_normatividad_sucesiones.txt", "03_politica_interna_simulada.txt"})

    def test_lista_verificacion_devuelve_checklist_no_plazos_solos(self):
        q = "Prepara una lista de verificación para revisar una solicitud incompleta relacionada con una sucesión, usando únicamente el repositorio. Separa documentos para acreditar la calidad de heredero, actuación ante varios herederos y gestión de documentos faltantes. Indica la fuente de cada punto."
        self.assertTrue(self.agente._es_lista_verificacion_sucesion(q))
        respuesta, _ = self.agente._respuesta_lista_verificacion_sucesion()
        self.assertIn("1. Acreditación de la calidad de heredero", respuesta)
        self.assertIn("2. Varios herederos sin partición", respuesta)
        self.assertIn("3. Gestión de documentos faltantes", respuesta)

    def test_plazo_judicial_prioriza_plazo_del_oficio(self):
        q = "Una solicitud de una autoridad judicial fija un plazo específico para responder. Según la política interna simulada, ¿cómo debe tratarse ese plazo? Indica qué información del repositorio permite responder y qué no puede concluirse."
        self.assertTrue(self.agente._es_plazo_autoridad_judicial(q))
        respuesta, _ = self.agente._respuesta_plazo_autoridad_judicial()
        self.assertIn("plazo indicado en el oficio", respuesta)
        self.assertIn("no debe sustituirse automáticamente", respuesta.lower())
        self.assertIn("sin revisar el oficio", respuesta)

    def test_acciones_registro_proteccion_separa_documentos_de_heredero(self):
        q = "¿Qué acciones de registro y protección menciona el repositorio cuando se entrega información de una persona fallecida? Enumera solo las acciones documentadas y cita la fuente."
        self.assertTrue(self.agente._es_acciones_registro_proteccion(q))
        respuesta, fuentes = self.agente._respuesta_acciones_registro_proteccion()
        self.assertIn("identidad del solicitante", respuesta)
        self.assertIn("finalidad, seguridad y confidencialidad", respuesta)
        self.assertIn("son una cuestión distinta", respuesta)
        self.assertEqual([f for _, f in fuentes], ["02_proteccion_datos.txt"])

    def test_nuevas_consultas_se_resuelven_antes_de_busqueda_rag(self):
        casos = [
            ("Resume las condiciones documentales para tramitar una solicitud relacionada con una sucesión. Separa los documentos generales, los requisitos cuando existe un proceso sucesoral y las condiciones aplicables si hay varios herederos sin partición. Indica la fuente de cada punto.", "registro civil de defunción"),
            ("Compara el Documento de referencia sobre sucesiones con la Política interna simulada de atención a solicitudes de sucesión. ¿Qué aporta cada documento y qué aspectos no deben confundirse?", "Comparación"),
            ("Prepara una lista de verificación para revisar una solicitud incompleta relacionada con una sucesión, usando únicamente el repositorio. Separa documentos para acreditar la calidad de heredero, actuación ante varios herederos y gestión de documentos faltantes. Indica la fuente de cada punto.", "Lista de verificación"),
            ("Una solicitud de una autoridad judicial fija un plazo específico para responder. Según la política interna simulada, ¿cómo debe tratarse ese plazo? Indica qué información del repositorio permite responder y qué no puede concluirse.", "plazo indicado en el oficio"),
            ("¿Qué acciones de registro y protección menciona el repositorio cuando se entrega información de una persona fallecida? Enumera solo las acciones documentadas y cita la fuente.", "identidad del solicitante"),
            ("¿Qué artículo exacto de una ley colombiana vigente autoriza entregar al heredero todos los movimientos bancarios de la persona fallecida? Responde solo si el repositorio lo sustenta.", "no infiero una autorización"),
        ]
        for pregunta, esperado in casos:
            with self.subTest(pregunta=pregunta[:45]):
                agente = AgenteJuridico.__new__(AgenteJuridico)
                agente._clasificar_mensaje = lambda p: "consulta"
                agente.recuperar_fragmentos = lambda p: (_ for _ in ()).throw(AssertionError("La regla determinista debió responder antes de RAG"))
                respuesta, fuentes = agente.generar_respuesta(pregunta)
                self.assertIn(esperado.lower(), respuesta.lower())
                self.assertTrue(fuentes)

    def test_movimientos_bancarios_no_infiere_autorizacion(self):
        q = "¿Qué artículo exacto de una ley colombiana vigente autoriza entregar al heredero todos los movimientos bancarios de la persona fallecida? Responde solo si el repositorio lo sustenta."
        self.assertTrue(self.agente._es_consulta_movimientos_bancarios_sin_soporte(q))
        respuesta, fuentes = self.agente._respuesta_movimientos_bancarios_sin_soporte()
        self.assertIn("no permite concluir", respuesta.lower())
        self.assertIn("no infiero una autorización", respuesta.lower())
        self.assertEqual({f for _, f in fuentes}, {"01_normatividad_sucesiones.txt", "02_proteccion_datos.txt"})

if __name__ == "__main__":
    unittest.main(verbosity=2)

class TestOchoPruebasUsuario(unittest.TestCase):
    """Pruebas deterministas para las ocho consultas funcionales del usuario."""
    def setUp(self):
        self.agente = AgenteJuridico.__new__(AgenteJuridico)

    def test_P01_sancion_sin_inferir(self):
        q = ("¿Qué sanción específica contempla el repositorio por entregar información sucesoral "
             "sin verificar los documentos? Si no existe información suficiente para identificarla, "
             "declara expresamente ese vacío sin afirmar que no se aplica ninguna sanción.")
        r, f = self.agente.generar_respuesta(q)
        self.assertIn("no identifican una sanción específica", r.lower())
        self.assertIn("no permite concluir que no exista", r.lower())
        self.assertNotIn("se puede inferir que la entidad", r.lower())
        self.assertTrue(f)

    def test_P03_cuatro_apartados_y_plazos_diferenciados(self):
        q = ("Resume las precauciones para atender solicitudes relacionadas con sucesiones y obligaciones crediticias "
             "en cuatro apartados: acreditación del solicitante, restricciones de acceso a la información, "
             "registro de la entrega y gestión de documentos faltantes. Para cada apartado, indica qué está "
             "respaldado por el repositorio y qué información no está disponible.")
        r, f = self.agente.generar_respuesta(q)
        for expected in ("Acreditación del solicitante", "Restricciones de acceso", "Registro de la entrega", "Gestión de documentos faltantes"):
            self.assertIn(expected.lower(), r.lower())
        self.assertIn("cinco (5) días hábiles", r)
        self.assertIn("quince (15) días hábiles", r)
        self.assertTrue(f)

    def test_P07_no_infiere_existencia_de_creditos(self):
        q = ("Una persona que no acredita parentesco ni representación solicita datos sobre las obligaciones "
             "crediticias de una persona fallecida. ¿Qué permite concluir la documentación disponible y qué "
             "información falta para resolver el caso?")
        r, f = self.agente.generar_respuesta(q)
        self.assertIn("no permiten confirmar si la persona fallecida tenía obligaciones crediticias", r.lower())
        self.assertNotIn("lo que sugiere que la persona fallecida tenía obligaciones crediticias", r.lower())
        self.assertTrue(f)

    def test_P09_separa_campos_de_documentos_de_acreditacion(self):
        q = ("¿Qué información debe quedar registrada cuando se entrega información sobre una obligación crediticia "
             "de una persona fallecida? Separa los campos expresamente respaldados por los documentos de aquellos "
             "que no aparecen en el repositorio. No infieras que los documentos necesarios para acreditar al "
             "solicitante son, por sí solos, los campos de trazabilidad de la entrega.")
        r, f = self.agente.generar_respuesta(q)
        self.assertIn("identidad del solicitante", r.lower())
        self.assertIn("alcance de la información entregada", r.lower())
        self.assertIn("no son, por ese solo hecho, campos de trazabilidad", r.lower())
        self.assertTrue(f)

    def test_P10_pregunta_ambigua_pide_aclaracion(self):
        r, _ = self.agente.generar_respuesta("¿Qué puedo hacer cuando tengo que pagar un crédito que no genere?")
        self.assertIn("podrías indicar", r.lower())

    def test_P11_sucesion_sola_pide_precision(self):
        r, _ = self.agente.generar_respuesta("Sucesión")
        self.assertIn("qué aspecto de la sucesión deseas consultar", r.lower())

    def test_P12_saludo_como_vas(self):
        r, _ = self.agente.generar_respuesta("¿Cómo vas?")
        self.assertIn("es un gusto saludarte", r.lower())

    def test_P13_escalamiento_no_inventa_canal(self):
        r, _ = self.agente.generar_respuesta("¿Con quién me puedo apoyar si no tienes la información necesaria para responder mi consulta?")
        self.assertIn("no identifica un canal institucional específico", r.lower())

if __name__ == "__main__":
    unittest.main()
