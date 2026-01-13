# AGENTS.md — State Engine (PA-first) + Event Scorer (telemetry-first)

## Propósito
Este repositorio implementa un **State Engine** para trading discrecional / semisistemático basado en **Price Action (PA)**.  
Su objetivo es **clasificar el estado estructural del mercado en H1** y **habilitar/prohibir familias de setups** mediante reglas explícitas (`ALLOW_*`), reduciendo errores estructurales, ruido cognitivo y sobre–operación.

El sistema **NO predice dirección next-bar**.  
Gobierna **cuándo existen condiciones estructurales válidas** para que un trade direccional sea considerado.

## Decisiones de diseño (NO negociables)

### Separación estricta de capas
- **ML = percepción** (clasificar estado / medir edge).
- **Reglas = decisión** (gating / habilitación).
- **Ejecución = determinista**, externa al ML.
- El ML **no toma decisiones operativas**.

### Métrica principal
- **Expectancy y drawdown** condicionados por **estado** y **familia**.
- Accuracy / F1 / AUC son **diagnósticos**, no objetivos.

### TRANSICIÓN es crítica
- `TRANSITION` **prohíbe swing direccional por defecto**.
- Solo habilita tácticos de tipo **failure / reclaim**.
- Operar direccionalmente en `TRANSITION` sin evidencia explícita es **error estructural**.

### Complejidad controlada
- H1: máx. ~10–12 features **PA-first**.
- Si se requieren más, la definición del problema está mal.
- M5 puede ser más rico, pero **siempre condicionado por H1**.

---

## Arquitectura del repositorio

configs/
symbols/
_template.yaml
XAUUSD.yaml

scripts/
train_state_engine.py
train_event_scorer.py
run_pipeline_backtest.py
run_walkforward_backtest.py
run_batch_walkforward.py
run_som_backtest_batch.py
summarize_event_scorers.py
watchdog_state_engine.py
watchdog_state_engine_whatsapp.py

state_engine/
backtest.py
config_loader.py
events.py
features.py
gating.py
labels.py
model.py
mt5_connector.py
pipeline.py
scoring.py
session.py
sweep.py
transition_shadow.py
walkforward.py

tests/
test_config_loader.py
test_diagnostic_report.py
test_events.py
test_features.py
test_session_bucket.py
test_train_event_scorer.py
test_walkforward.py

---

## Scope reutilizable del repo previo
Se recicla **infraestructura**, no lógica:
- `MT5Connector` (OHLCV).
- Utilidades de calendario / sesión / timezone.
- Infra mínima de dataset (resampling, NaN, persistencia).
- Persistencia de modelos.

Todo lo demás es **legado conceptual**.

---

## State Engine (H1)

### Definición de estado
**Timeframe:** H1

**Ventanas fijas**
- `W = 24` velas (contexto)
- `N = 8` velas (reciente)

**Estados**
- **BALANCE**: rotación, aceptación bilateral.
- **TRANSITION**: intento de salida con aceptación incompleta o fallo.
- **TREND**: migración sostenida con aceptación.

**Regla maestra**  
El estado se define por **comportamiento agregado**, no por patrones aislados.

### Normalización por volatilidad
- `ATR_W = ATR(t-W .. t)`
- `ATR_N = ATR(t-N .. t)`

Contexto → `ATR_W`  
Ruptura / reciente → `ATR_N`

### Variables PA-first (≤ t)
- NetMove (diagnóstico)
- Path
- Efficiency Ratio (ER)
- Range_W
- CloseLocation
- BreakMag
- ReentryCount
- InsideBarsRatio
- SwingCount
- Pendientes opcionales (ER, Range)

---

## Modelo State Engine
- **Modelo:** LightGBM multiclass
- **Outputs:** `state_hat`, `margin = P1 − P2`, probas (solo reporting)

Notas:
- No se asume calibración perfecta.
- Gating usa `state_hat + margin`, no probas crudas.

---

## Gating determinista (ALLOW_*)
- Lógica explícita, auditable.
- No entrenable.

Ejemplos:
- `ALLOW_trend_pullback`: TREND & margin ≥ umbral
- `ALLOW_balance_fade`: BALANCE & margin ≥ umbral
- `ALLOW_transition_failure`: TRANSITION + condiciones explícitas

**Regla crítica**  
`TRANSITION` prohíbe swing direccional por defecto.

---

## Event Scorer (M5) — Edge, no señal

### Rol
Medir **edge relativo** de eventos M5 condicionado por contexto H1.

### Input
- Eventos M5 (proposal engine).
- Contexto H1: `state_hat_H1`, `margin_H1`, `ALLOW_*`.

### Output
- `edge_score ∈ [0,1]` (ranking / telemetría).

**Prohibiciones**
- NO señales.
- NO trading.
- NO SL/TP.
- NO decisiones operativas.

---

## Filosofía de eventos M5

### detect_events (proposal engine)
- Prioriza recall.
- Genera eventos solo si `ALLOW_*` activo.
- Adjunta features de fuerza/contexto.

### label_events
- Triple-barrier proxy → `r_outcome` continuo.
- Label binario derivado con umbral configurable.
- Conservador en empates TP/SL.

---

## Puente H1 → M5 (causal)
- Solo último H1 **cerrado**.
- Implementación: `shift(1)` + `merge_asof(backward)`.
- Prohibido usar H1 en formación (leakage grave).

---

## Entrenamiento Event Scorer

- Modelos LightGBM binarios por familia.
- Fallback a modelo global si muestras insuficientes.

**Métricas diagnósticas**
- `lift@K`, `r_mean@K`, `precision@K`.
- Breakdown por familia, estado, bins de margin.

---

## Configuración por símbolo y modos

Ubicación:
configs/symbols/
_template.yaml
XAUUSD.yaml

---

Modos soportados:
- **default/original** (sin config)
- **production**
- **research**

### Regla de oro
Si `research.enabled == false`:
- comportamiento
- métricas
- artefactos  
deben ser **equivalentes al modo original**.

---

## Research mode (opt-in, diagnóstico)

Research **NO mejora producción automáticamente**.  
Sirve para **explorar edge y score-shape**.

### Capacidades (solo research)
- Features exógenas:
  - session_bucket
  - hour_bucket
  - trend_context_D1
  - vol_context
- Exploración multi-k:
  - `k_bars_grid: [12,24,36]`
  - artefactos con sufijo `_k{K}`

### Guardrails diagnósticos
- `RESEARCH_OK`
- `RESEARCH_UNSTABLE`
- `RESEARCH_OVERFIT_SUSPECT`

No afectan decisiones ni entrenamiento productivo.

**Nota D1**
- D1 basado en día calendario puede ser inestable según timezone.
- Research debe advertir si aplica.

---

## Backtesting
- Determinista (M5).
- Entry: next_open.
- SL/TP con OHLC (SL primero).
- Fees/slippage configurables.
- Walk-forward mensual.
- `allow_overlap = False`.

---

## Resultado esperado
Un sistema que:
- Clasifica estado con robustez.
- Habilita/prohíbe familias explícitamente.
- Genera abundancia controlada de oportunidades.
- Prioriza edge, supervivencia y disciplina.
- Reduce errores estructurales.

**Regla final:**  
Convertir `edge_score` en señal directa **rompe el diseño del sistema**.
