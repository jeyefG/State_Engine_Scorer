Eres Codex. Trabajas sobre el repo “state_engine”. Implementa (1) Event Scorer (ML) y (2) Backtesting + Param Sweep + UI/Logging. No rompas compatibilidad. No mezcles responsabilidades. No conviertas outputs de modelos en señales. Mantén simpleza.

================================================================
SEMÁNTICA INNEGOCIABLE
================================================================
- State Engine y gating operan SOLO en H1.
- Event Scorer opera SOLO en M5. Output = edge_score (información, no acción).
- Backtester consume SOLO señales deterministas (Signal). No consume outputs “crudos” de modelos como acciones.
- Prohibido leakage: features <= t; labels usan (t+1..t+K).
- Contexto H1 aplicado a M5 debe ser el último H1 CERRADO: shift(1) + merge_asof.
- Prohibido ejecutar trades “solo” por edge_score. Señal requiere:
  allow activo AND evento detectado AND edge_score > threshold AND trigger mecánico explícito.

================================================================
CONTRATO DE TIMEFRAMES
================================================================
- State Engine y gating: H1 only.
- Event Scorer: M5 only.
- Backtesting: M5 only (fills conservadores).
- Prohibido recalcular state/allow en M5.

================================================================
EVENTOS (CANDIDATOS) — DEFINICIÓN MINIMALISTA
================================================================
El scorer NO corre sobre todas las velas. Corre SOLO cuando hay un evento mínimo (candidato).
Familias de eventos (pocas):
- E_BALANCE_FADE
- E_BALANCE_REVERT
- E_TRANSITION_TEST
- E_TRANSITION_FAILURE
- E_TREND_PULLBACK
- E_TREND_CONTINUATION
Si existen ALLOW_* adicionales, mapear a familia existente o crear una nueva SOLO si es inevitable.

Los detectores deben ser simples, causales, baratos. El detector solo dice “aquí pasó algo”.
El ML decide calidad/edge.

================================================================
PUENTE H1 → M5 (OBLIGATORIO)
================================================================
1) Calcular state_hat, margin y ALLOW_* en H1.
2) Aplicar shift(1) al contexto H1 (causalidad).
3) Merge_asof (backward) o equivalente para forward-fill hacia M5.
Cada fila M5 debe tener: features M5 + state_hat_H1 + margin_H1 + ALLOW_*.

Sanity check: ALLOW_* no debe cambiar dentro de la misma hora M5.

================================================================
ETIQUETADO (CRÍTICO)
================================================================
El etiquetado NO representa la operativa real. Es un instrumento de medición.
Definir un protocolo mecánico mínimo (proxy) para outcome, simétrico y reproducible:

Para cada evento en tiempo t:
- entry_price = next open (default)
- SL_proxy = ATR_short * sl_mult (configurable)
- TP_proxy = R * SL_proxy (configurable)
- ventana futura = K velas M5 (configurable)

Label:
- y = 1 si TP_proxy se alcanza ANTES que SL_proxy dentro de K
- y = 0 si SL_proxy se alcanza antes o si no se alcanza TP en K

No usar features futuras. No incluir columnas que codifiquen el futuro.

================================================================
FEATURES (MÍNIMAS, ROBUSTAS)
================================================================
Microestructura M5 (últimas N velas):
- returns (1,3)
- volatilidad realizada
- ATR corto (si no existe, calcular)
- rango/cuerpo/mechas (ratios)
- overlap/chop proxy
- ROC simple

Contexto:
- state_hat one-hot
- margin
- family_id one-hot
- ALLOW_* (binarias)

================================================================
MODELO (EVENT SCORER)
================================================================
- LightGBM o CatBoost, clasificador binario.
- Un solo modelo global con family_id como feature (simplicidad).
- Split temporal (NO shuffle).
- Calibración de probabilidades (Platt o Isotonic).
- Guardar modelo + calibrador en models/.

Output del scorer: edge_score (calibrado). Nada más.

================================================================
FRONTERA OUTPUT ↔ EJECUCIÓN
================================================================
Prohibido convertir edge_score directamente en BUY/SELL.
La señal se construye FUERA del modelo:

EJECUTO ⇔
  allow activo
  AND evento detectado
  AND edge_score > threshold_familia (o threshold global)
  AND trigger mecánico explícito

Trigger mecánico debe ser simple (no agenda grande), por ejemplo:
- LONG: break del high de vela evento/confirmación
- SHORT: break del low

================================================================
BACKTESTING (NUEVO)
================================================================
Implementar backtester determinista (M5) con fills conservadores:
- Entry: next_open (default).
- SL/TP evaluado con OHLC.
- Si SL y TP ocurren en la misma vela: resolver conservador (SL primero).
- Config: allow_overlap=False por defecto; max_holding_bars configurable; fee/slippage.

El backtester consume una lista de Signal (no raw scores) y produce:
- trades.csv
- equity.csv
- metrics dict (global + por familia + por bins edge_score)

Métricas mínimas:
- n_trades, win_rate, avg_win, avg_loss, expectancy
- profit_factor
- max_drawdown y duración
- exposure
- métricas por family_id
- métricas por bins de edge_score

Guardar siempre:
- events.csv (candidatos + contexto + edge_score + label si existe)
- signals.csv (señales finales)
- trades.csv
- equity.csv

================================================================
PARAM SWEEP (2–3 PARÁMETROS, SIMPLE)
================================================================
Implementar sweep liviano (solo si flag --sweep):
Iterar combinaciones pequeñas (<= 64 combos) sobre:
A) threshold_edge (global o por familia; si por familia es complejo, usar global)
B) K (ej: 12, 24)
C) R (ej: 0.8, 1.0, 1.2)  [o sl_mult, pero no ambos si ya son 3 parámetros]

Por cada combinación:
- construir señales
- correr backtest
- guardar row en sweep_results.csv con params + métricas clave

Mostrar Top 10 por profit_factor (y desempate por max_drawdown menor o equity_final mayor).
Mostrar Bottom 10.

================================================================
CAMBIOS CONCRETOS EN EL REPO
================================================================
1) state_engine/events.py
   - EventFamily
   - detect_events(df_m5_ctx) -> events_df
   - label_events(events_df, df_m5, K, R, sl_mult) -> events_df con y

2) state_engine/scoring.py
   - EventScorerConfig
   - FeatureBuilder
   - EventScorer fit/predict_proba/save/load
   - calibración

3) state_engine/backtest.py
   - Signal, Trade, BacktestConfig
   - run_backtest + compute_metrics

4) (Opcional liviano) state_engine/sweep.py
   - run_param_sweep(pipeline_fn, grid) -> results_df

5) scripts/train_event_scorer.py
   - Entrena scorer con split temporal, calibración, guarda modelo.
   - Logs: conteos eventos, distribución labels, métricas.

6) scripts/run_pipeline_backtest.py  (script principal)
   - Carga datos H1/M5
   - Corre/carga State Engine H1
   - Gating H1
   - Bridge H1→M5 (shift+merge_asof)
   - Detect events
   - Carga scorer y calcula edge_score (si no hay scorer, edge_score=0.5 con WARNING)
   - Construye señales (threshold + trigger mecánico + SL/TP proxy)
   - Corre backtest
   - Si --sweep, corre sweep sobre 2–3 params
   - Guarda CSVs y muestra report

En scripts nuevos:
- --end default = today + 1 day
- NO required=True

================================================================
UI / LOGGING (OBLIGATORIO, TEXTO)
================================================================
Imprimir reporte con secciones fijas:

[DATA]
- filas H1 / filas M5, rango fechas

[STATE ENGINE H1]
- distribución state_hat, percentiles margin

[GATING]
- % tiempo por ALLOW_ (H1), gating.tail(5)

[EVENTS]
- conteo por family_id, por side, tasa diaria

[SCORER]
- stats edge_score (mean, p50, p90) global y por familia; warning si fallback

[SIGNALS]
- conteo por familia tras threshold
- top 10 señales (time, family, side, edge, entry, sl, tp)

[BACKTEST]
- métricas globales y por familia
- top 10 trades y peores 10

[SAVE]
- rutas de CSVs generados

================================================================
SANITY CHECKS
================================================================
- ALLOW_* constante dentro de hora M5.
- Contexto H1 aplicado a M5 usa shift(1).
- entry_time > signal_time (next_open).
- Resolver SL/TP intrabar de forma conservadora.
- Si no hay eventos, terminar limpio con warning.

================================================================
ENTREGABLE
================================================================
- Código end-to-end funcional.
- Ejemplos de comandos.
- Mantener estilo del repo.
- NO romper scripts existentes.

Ahora implementa exactamente esto.
