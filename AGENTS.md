# AGENTS.md — State Engine (PA-first)

## Propósito
Este repositorio implementa un State Engine para trading discrecional / semisistemático basado en Price Action (PA).
El objetivo es **clasificar el estado estructural del mercado en H1** y **habilitar o prohibir familias de setups**
mediante reglas explícitas (ALLOW_*), reduciendo errores estructurales, ruido cognitivo y sobre–operación.

El sistema **NO predice dirección next-bar**.
Gobierna **cuándo existen condiciones estructurales válidas** para que un trade direccional sea considerado.

================================================================
Decisiones de diseño (no negociables)
================================================================

Separación estricta de capas:

- ML = percepción (clasificar estado / medir edge).
- Reglas = decisión (gating / habilitación).
- Ejecución = determinista, externa al ML.

El ML **no toma decisiones operativas**.

Métrica principal:

- Expectancy y drawdown **condicionados por estado y familia de setup**.
- Accuracy / F1 / AUC son métricas **diagnósticas**, no objetivos.

TRANSICIÓN es crítica:

- TRANSICIÓN **prohíbe swing direccional por defecto**.
- Solo habilita tácticos de tipo failure / reclaim.
- Operar direccionalmente en TRANSICIÓN sin evidencia explícita es considerado error estructural.

Complejidad controlada:

- H1: máx. ~10–12 features PA-first.
- Si se requieren más, la definición del problema está mal.
- M5 puede ser más rico, pero siempre condicionado por H1.

================================================================
Scope reutilizable del repo previo
================================================================

Se recicla **solo infraestructura**, no lógica:

- MT5Connector (OHLCV).
- Utilidades limpias de calendario / sesión / timezone.
- Infra mínima de dataset (resampling, NaN, persistencia).
- Persistencia de modelos.

Todo lo demás se considera legado conceptual.

================================================================
Definición de estado (H1)
================================================================

Timeframe: H1

Ventanas fijas:

- W = 24 velas H1 (contexto)
- N = 8 velas H1 (reciente)

Estados:

- BALANCE:
  Rotación, baja direccionalidad neta, aceptación bilateral.
- TRANSICIÓN:
  Intento de salida con aceptación incompleta o fallo.
  Estado inherentemente peligroso para swing direccional.
- TENDENCIA:
  Migración sostenida con aceptación.
  Retrocesos ordenados, direccionalidad clara.

Regla maestra:
El estado se define por **comportamiento agregado en ventanas fijas**,
no por velas individuales ni patrones aislados.

================================================================
Normalización por volatilidad
================================================================

ATR_W = ATR(t-W .. t)
ATR_N = ATR(t-N .. t)

- Métricas de contexto → normalizar por ATR_W
- Métricas recientes / ruptura → normalizar por ATR_N

================================================================
Variables PA-first (≤ t)
================================================================

NetMove (diagnóstico):
|C_t − C_{t-W}| / ATR_W

Path:
∑ |C_i − C_{i−1}| / ATR_W

Efficiency Ratio (ER):
|C_t − C_{t-W}| / ∑ |C_i − C_{i−1}|

Range_W:
(H_W − L_W) / ATR_W

CloseLocation:
(C_t − L_W) / (H_W − L_W) ∈ [0,1]
(si H_W == L_W → 0.5)

BreakMag:
max(0, |C_t − clamp(C_t, L_W, H_W)|) / ATR_N

ReentryCount:
Cantidad de transiciones outside → inside en ventana N

InsideBarsRatio:
(# inside bars en N) / N

SwingCount:
Pivots confirmados (high + low) en W
(pivots usan solo info hasta t−1)

Pendientes (opcionales):
ERSlope, RangeSlope

================================================================
Bootstrap inicial (NO verdad de mercado)
================================================================

El bootstrap existe **solo para arrancar el sistema**.
NO se optimiza por accuracy.

Regla crítica:
Las variables usadas para definir el bootstrap **NO pueden ser reutilizadas**
como features core del modelo sin modificación explícita.

Política:

- ER y NetMove son métricas diagnósticas.
- Si se usan en bootstrap → NO se usan como features core.
- Si se incluyen como features → deben salir del bootstrap o degradarse.
- Prioridad conservadora: TRANSICIÓN > TENDENCIA > BALANCE.

================================================================
Modelo principal — StateEngine (H1)
================================================================

Modelo:
LightGBM multiclass (BALANCE / TRANSICIÓN / TENDENCIA)

Outputs:

- state_hat
- margin = P(top1) − P(top2)
- probas (solo reporting)

Notas:

- No se asume calibración perfecta.
- El gating usa state_hat + margin, no probas crudas.

================================================================
Gating determinista (ALLOW_*)
================================================================

ALLOW_* **no se entrena**.
Es lógica explícita, revisable y auditable.

Ejemplos:

- ALLOW_trend_pullback:
  state_hat == TENDENCIA y margin ≥ 0.15

- ALLOW_trend_continuation:
  state_hat == TENDENCIA y margin ≥ 0.15

- ALLOW_balance_fade:
  state_hat == BALANCE y margin ≥ 0.10

- ALLOW_transition_failure:
  state_hat == TRANSICIÓN
  margin ≥ 0.10
  BreakMag ≥ 0.25
  ReentryCount ≥ 1

Regla explícita:
Si state_hat == TRANSICIÓN → swing direccional prohibido por defecto.

================================================================
Event Scorer (M5) — Edge, no señal
================================================================

Rol:
Medir **edge relativo** de eventos M5 **condicionado por contexto H1**.

Input:

- Eventos M5 (propuestos, no filtrados agresivamente).
- Contexto H1: state_hat_H1, margin_H1, ALLOW_*.

Output:

- edge_score ∈ [0,1] (ranking)

Principios:

- El Event Scorer:
  - NO genera señales
  - NO ejecuta trades
  - NO define SL / TP
- Convierte reglas duras en **ranking probabilístico**.
- Permite abundancia de candidatos y selección posterior.

================================================================
Filosofía de eventos M5
================================================================

detect_events:

- Funciona como **proposal engine**, no como filtro duro.
- Genera candidatos SOLO cuando el ALLOW correspondiente está activo.
- Prioriza recall sobre precisión.
- Adjunta features de contexto/fuerza del evento (ATR, momentum, compresión, ubicación).

label_events:

- Usa triple-barrier proxy con r_outcome continuo.
- Etiqueta binaria deriva de r_outcome con umbral configurable.
- Conservador en empates TP/SL.

================================================================
Puente H1 → M5 (causal)
================================================================

El contexto H1 se aplica a M5 usando SOLO el último H1 CERRADO.

Implementación:
shift(1) + merge_asof(direction="backward")

Está prohibido usar la vela H1 en formación.
Cualquier violación se considera leakage.

================================================================
Entrenamiento Event Scorer
================================================================

- Se entrena un modelo global y, opcionalmente, modelos por familia.
- Familias con pocas muestras:
  - NO entrenan modelo propio.
  - Usan fallback al modelo global.
- Métricas clave:
  - precision@K
  - lift@K
  - r_mean@K
- Evaluación por:
  - familia
  - bins de margin_H1

Accuracy/AUC solo diagnósticos.

================================================================
Backtesting
================================================================

Backtesting determinista (M5):

- Entry: next_open
- SL/TP evaluado con OHLC
- Si SL y TP ocurren en la misma vela → SL primero (conservador)
- Fees y slippage configurables
- allow_overlap = False por defecto
- Walk-forward mensual

================================================================
Resultado esperado
================================================================

Un sistema que:

- Clasifica estado con robustez.
- Habilita/prohíbe familias explícitamente.
- Genera abundancia controlada de oportunidades.
- Prioriza edge, supervivencia y disciplina sobre actividad.
- Reduce decisiones impulsivas y errores estructurales.

================================================================
Notas finales
================================================================

El State Engine gobierna **cuándo pensar en operar**.
El Event Scorer mide **qué eventos merecen atención relativa**.
La ejecución queda fuera del ML.

Cualquier intento de convertir edge_score en señal directa
rompe el diseño del sistema.
