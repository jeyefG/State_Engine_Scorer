Eres Codex. Objetivo: extender el watchdog H1 existente para que, cuando haya ALLOW_* activo en la última H1 cerrada, muestre (opcionalmente) un resumen de oportunidades del Event Scorer (M5) como TELEMETRÍA (no señal). No romper arquitectura: el watchdog NO ejecuta trades, NO genera señales, NO crea órdenes. Solo reporta contexto H1 + top eventos M5 con edge_score.

================================================================
SEMÁNTICA INNEGOCIABLE
================================================================
- Watchdog es H1-first: su gatillo principal es “nueva vela H1 cerrada”.
- El Event Scorer es M5-only: output edge_score (información, no acción).
- Prohibido convertir edge_score en BUY/SELL o “señal”.
- Prohibido usar H1 en formación: usar cutoff = server_now.floor("h") y ohlcv_h1 < cutoff.
- Prohibido leakage: M5 evaluado debe ser <= cutoff.
- El scorer se evalúa SOLO si allow_any=True en la última H1 cerrada (reduce ruido).

================================================================
CAMBIO FUNCIONAL
================================================================
Cuando el watchdog detecta una nueva H1 cerrada y allow_any=True en el último índice:
- además del summary H1 actual (state_hat/margin/rules fired),
- imprimir un bloque adicional: “Event Scorer (M5) snapshot” que contenga:
  - ventana M5 evaluada (ej: last N minutos hasta cutoff)
  - conteo de eventos por familia detectados en esa ventana
  - top K eventos por edge_score con:
    timestamp_event, family_id, side, edge_score
    (opcional) entry_proxy_time (next M5 open) y entry_proxy_price
- Si no hay eventos en ventana: reportar “no events detected”
- Si no hay scorer disponible: reportar “scorer not available” (sin error fatal)

NO generar files (signals.csv/trades.csv) desde watchdog.
NO backtesting desde watchdog.

================================================================
ARGUMENTOS CLI NUEVOS (watchdog)
================================================================
Agregar flags (todos opcionales):
- --enable-scorer (bool) default False
- --scorer-dir PATH (default: args.model_dir o PROJECT_ROOT/models)
- --scorer-template STR (default "{symbol}_event_scorer.pkl" o el nombre real usado en repo)
- --m5-lookback-min INT (default 180)
- --top-events INT (default 5)
- --min-edge-score FLOAT (default None; si se setea, filtrar eventos bajo ese score en el display)

Mantener compatibilidad: si no se usa --enable-scorer, watchdog se comporta igual que hoy.

================================================================
INTEGRACIÓN TÉCNICA (USAR MÓDULOS EXISTENTES DEL REPO)
================================================================
Reutilizar los módulos ya implementados:
- state_engine.events: detect_events(...) para obtener events_df
- state_engine.scoring: EventScorer.load(...) y predict_proba(...) para edge_score
- bridge H1→M5: usar la misma lógica causal del repo (shift(1)+merge_asof(backward)).
  Si ya existe helper, úsalo. Si no existe, implementar localmente dentro del watchdog de forma simple y auditable.

Data requirements:
- Descargar M5 para la ventana: [cutoff - m5_lookback_min, cutoff]
- Descargar H1 lookback ya existente (para state/gating).
- Construir contexto H1 (outputs+gating) y aplicarlo a M5 (último H1 cerrado) antes de detectar eventos.

Regla: eventos M5 deben detectar solo cuando ALLOW_* correspondiente está activo en M5 (por forward-fill del allow desde H1).

================================================================
UI / LOGGING (TEXTO O RICH)
================================================================
Si Rich está disponible, agregar sección debajo del summary actual:

=== Event Scorer (M5) Snapshot ===
Window: 2025-.. .. -> cutoff
Events: total=.. | by_family={...}
Top events:
1) ts=... family=E_... side=LONG edge=0.71 entry_proxy=... price=...
...

Si no hay Rich, imprimir lo mismo en texto plano.

IMPORTANTE:
- No imprimir métricas de entrenamiento (accuracy/f1) en modo operación, salvo que ya exista y sea requerido.
- Mantener el output enfocado: estado/margin/rules + snapshot scorer.

================================================================
MANEJO DE ERRORES
================================================================
- Si falla descarga M5: warning y continuar.
- Si scorer file no existe: warning “scorer not available” y continuar.
- Si detect_events no encuentra eventos: imprimir “no events detected”.
- Nunca crash del watchdog por scorer.

================================================================
LISTA DE TAREAS CONCRETAS
================================================================
1) Modificar parse_args() para agregar flags.
2) Implementar load_event_scorer(symbol, scorer_dir, scorer_template) con safe_symbol.
3) Implementar fetch_m5(symbol, start, end) usando MT5Connector (crear método si no existe; si ya existe, usarlo).
4) Implementar bridge H1→M5:
   - tomar outputs H1 + gating H1
   - shift(1)
   - merge_asof hacia index M5
5) En loop por símbolo:
   - si allow_any del último H1 es True y --enable-scorer:
       - correr snapshot M5 y renderizar
6) Mantener comportamiento actual si --enable-scorer=False.

================================================================
ENTREGABLE
================================================================
- Watchdog actualizado con scorer snapshot.
- Sin cambios al pipeline de training/backtest.
- Sin romper scripts existentes.
Implementa ahora.
