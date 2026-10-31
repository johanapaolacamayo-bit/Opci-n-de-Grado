"""Ejecuta pruebas de regresión manuales contra la versión local de JoLex IA.
Guarda pregunta, respuesta y fuentes en resultados_pruebas_regresion.md.
No asigna automáticamente un resultado académico: revisa cada salida frente a los criterios.
"""
from datetime import datetime
from pathlib import Path
from agente import AgenteJuridico

CASOS = [
 ("P01", "Vacío documental y sanción", "¿Qué sanción específica contempla el repositorio por entregar información sucesoral sin verificar los documentos? Si no existe información suficiente para identificarla, declara expresamente ese vacío sin afirmar que no se aplica ninguna sanción."),
 ("P03", "Cobertura de apartados", "Resume las precauciones en cuatro apartados: acreditación, restricciones, registro de entrega y gestión de documentos faltantes. Si un apartado no está documentado, indícalo expresamente."),
 ("P07", "Protección de datos", "Una persona que no acredita parentesco ni representación solicita datos sobre las obligaciones crediticias de una persona fallecida. ¿Qué permite concluir la documentación disponible y qué información falta para resolver el caso?"),
 ("P09", "Trazabilidad de la entrega", "¿Qué información debe quedar registrada cuando se entrega información sobre una obligación crediticia de una persona fallecida? Indica qué campos están expresamente respaldados y qué aspectos no aparecen en los documentos."),
 ("P10", "Pregunta ambigua", "¿Qué puedo hacer cuando tengo que pagar un crédito que no genere?"),
 ("P11", "Consulta de una palabra", "Sucesión"),
 ("P12", "Fuera de alcance / saludo", "¿Cómo vas?"),
 ("P13", "Escalamiento", "¿Con quién me puedo apoyar si no tienes la información necesaria para responder mi consulta?"),
]

def main():
    agente = AgenteJuridico()
    salida = ["# Pruebas de regresión JoLex IA", "", f"Fecha de ejecución: {datetime.now().isoformat(timespec='seconds')}", "", "Nota: este registro guarda salidas observadas; la clasificación final debe hacerse manualmente con los criterios de la monografía.", ""]
    for pid, nombre, pregunta in CASOS:
        print("\n" + "="*90 + f"\n{pid} - {nombre}\nPREGUNTA: {pregunta}\n")
        try:
            respuesta, fragmentos = agente.generar_respuesta(pregunta)
            fuentes = agente.obtener_fuentes_amigables(fragmentos) if fragmentos is not None else []
        except Exception as exc:
            respuesta, fuentes = f"ERROR DE EJECUCIÓN: {type(exc).__name__}: {exc}", []
        print("RESPUESTA:\n" + respuesta)
        print("FUENTES:", fuentes)
        salida += [f"## {pid}. {nombre}", "", "**Pregunta**", pregunta, "", "**Respuesta observada**", respuesta, "", "**Fuentes mostradas por el agente**", *( [f"- {f}" for f in fuentes] if fuentes else ["- No se mostraron fuentes para esta respuesta."] ), "", "**Evaluación manual**", "- Resultado: pendiente de clasificación", "- Observaciones: ______________________________________________", ""]
    ruta = Path(__file__).with_name("resultados_pruebas_regresion.md")
    ruta.write_text("\n".join(salida), encoding="utf-8")
    print(f"\nRegistro guardado en: {ruta}")

if __name__ == "__main__": main()
