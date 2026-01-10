Eres Codex. Objetivo: extender el watchdog H1 existente para que, cuando haya ALLOW_* activo en la ÚLTIMA H1 CERRADA, muestre un resumen de oportunidades del Event Scorer (M5) como TELEMETRÍA (no señal). El watchdog NO ejecuta trades, NO genera señales, NO crea órdenes. Solo reporta contexto H1 + ranking de eventos M5 con edge_score. El Event Scorer se ejecuta únicamente bajo condición de ALLOW activo.

================================================================
SEMÁNTICA INNEGOCIABLE
================================================================
- El watchdog es H1-first: su gatillo principal es la detección de una nueva vela H1 CERRADA.
- El Event Scorer es M5-only y es PARTE OBLIGATORIA del output del watchdog
  CUANDO existe al menos un ALLOW_* activo en la ÚLTIMA H1 CERRADA.
- Si NO hay ALLOW_* activo en la última H1 cerrada, el Event Scorer NO se ejecuta y NO se muestra.
- El Event Scorer nunca corre fuera de un ALLOW activo.
- El output del Event Scorer es edge_score (telemetría), NO señal.
- Está prohibido convertir edge_score en BUY/SELL, trigger automático o acción ejecutable.
- Está prohibido usar H1 en formación o M5 posterior al cutoff (anti-leakage).

================================================================
CAMBIO FUNCIONAL
================================================================
Cuando el watchdog detecta una nueva H1 cerrada (last_idx):

- Si allow_any == False en last_idx:
    - Renderizar SOLO el summary H1:
        estado (state_hat), margin y rules fired.
    - NO ejecutar lógica M5.
    - NO cargar ni evaluar el Event Scorer.

- Si allow_any == True en last_idx:
    - Renderizar el summary H1.
    - Ejecutar de forma OBLIGATORIA un snapshot M5 del Event Scorer.
    - Mostrar un ranking de eventos M5 con edge_score.

================================================================
ARGUMENTOS CLI (watchdog)
================================================================
Agregar los siguientes flags (todos opcionales):

- --scorer-dir PATH
    Directorio donde se encuentran los modelos del Event Scorer
    (default: args.model_dir o PROJECT_ROOT/models)

- --scorer-template STR
    Plantilla del nombre del scorer
    (default: "{symbol}_event_scorer.pkl")

- --m5-lookback-min INT
    Ventana M5 a evaluar hacia atrás desde el cutoff
    (default: 180 minutos)

- --top-events INT
    Número máximo de eventos M5 a mostrar en el ranking
    (default: 5)

- --min-edge-score FLOAT
    Umbral mínimo para mostrar eventos en el ranking
    (default: None → no filtra)

NO existe flag --enable-scorer.
La ejecución del Event Scorer es condicional al ALLOW H1, no opcional.

================================================================
INTEGRACIÓN TÉCNICA (USAR MÓDULOS EXISTENTES)
================================================================
Reutilizar módulos del repo siempre que existan:

- state_engine.events.detect_events(...) → generar events_df (candidatos M5)
- state_engine.scoring.EventScorer.load(...) y predict_proba(...) → edge_score
- Puente H1→M5:
    Aplicar SIEMPRE el último H1 cerrado a M5 usando
    shift(1) + merge_asof(backward).

Data requirements:
- Descargar H1 con el lookback ya existente para state/gating.
- Descargar M5 en el rango:
    [cutoff − m5_lookback_min, cutoff]
- Aplicar contexto H1 (outputs + gating) a M5 ANTES de detectar eventos.
- Detectar eventos M5 SOLO cuando el ALLOW_* correspondiente está activo
  (forward-fill del allow desde H1).

================================================================
UI / LOGGING (OPERATIVO)
================================================================
Bajo el summary H1 existente, agregar:

=== Event Scorer (M5) Snapshot ===
Window: <start> → <cutoff>
Events detected: total=<n> | by_family={<family>: <count>, ...}

Top events (sorted by edge_score desc):
1) ts=<event_ts> | family=<E_*> | side=<LONG/SHORT> | edge=<0.xx>
   entry_proxy_time=<next_m5_open> | entry_proxy_price=<price>
...

Si no hay eventos en la ventana:
- Mostrar: “no events detected in M5 window”.

Si el scorer no está disponible:
- Mostrar: “scorer not available – cannot rank opportunities”.

No imprimir métricas de entrenamiento (accuracy, F1) en modo operación.
El foco del output es: contexto H1 + ranking M5.

================================================================
MANEJO DE ERRORES
================================================================
- Si falla la descarga M5 → warning y continuar.
- Si el archivo del scorer no existe o falla el load → warning y continuar.
- Nunca hacer crash del watchdog por el Event Scorer.
- Si allow_any=True y el scorer no está disponible:
    - Mostrar summary H1.
    - Mostrar bloque Event Scorer con mensaje de indisponibilidad.

================================================================
LISTA DE TAREAS CONCRETAS
================================================================
1) Extender parse_args() con los flags del scorer.
2) Implementar load_event_scorer(symbol, scorer_dir, scorer_template)
   usando safe_symbol.
3) Implementar fetch_m5(symbol, start, end) con MT5Connector
   (reusar método existente si ya está).
4) Implementar puente H1→M5:
   - outputs H1 + gating H1
   - shift(1)
   - merge_asof hacia índice M5
5) En el loop por símbolo:
   - Si allow_any == True en la ÚLTIMA H1 CERRADA:
       - ejecutar snapshot M5 del Event Scorer y renderizar
   - Si allow_any == False:
       - NO ejecutar scorer
6) Mantener intacto el pipeline de training y backtesting.

================================================================
ENTREGABLE
================================================================
- Watchdog actualizado con snapshot del Event Scorer condicionado por ALLOW.
- Sin generación de señales ni órdenes.
- Sin romper scripts existentes.
Implementa ahora.
