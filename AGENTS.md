# AGENTS.md — State Engine (PA-first, no Taylor/VWAP)

## Propósito
Este repositorio implementa un **State Engine** para trading discrecional/semisistemático basado en **Price Action (PA)**.
El objetivo es **clasificar el estado del mercado** y **habilitar/prohibir familias de setups** mediante reglas explícitas (`ALLOW_*`), reduciendo errores estructurales y sosteniendo estabilidad cognitiva.

Este sistema **no predice dirección**. Gobierna **cuándo existe un trade**.

---

## Decisiones de diseño (no negociables)
1) **Taylor/VWAP: excluidos del modelo**.  
   No se usan como features, señales ni contexto.  
   No se recicla lógica previa por conveniencia.
2) **ML = percepción** (clasificar estado). **Reglas = decisión** (gating).  
3) **Métrica principal**: expectancy y drawdown condicionados por estado y setup.  
   Accuracy/F1 son secundarias.
4) **Transición es crítica**: su función es **prohibir direccional**.
5) **Complejidad mínima**: máximo 10–12 features H1. Si se requieren más, la definición está mal.

---

## Scope reutilizable del repo previo
Se recicla **solo infraestructura**, no lógica:
- `MT5Connector`: descarga OHLCV por símbolo/timeframe.
- Utilidades de calendario/sesión (si existen y están limpias).
- Infra mínima de dataset (resampling, timezone, NaN handling).
- Persistencia de modelos (save/load).

Todo lo demás se considera legado y **no se integra**.

---

## Definición de estado (H1)
- **Timeframe**: H1  
- **Ventana fija**: `W = 24` velas H1

Estados:
- **BALANCE**: negociación de valor (consenso).
- **TRANSICIÓN**: intento de cambio sin aceptación.
- **TENDENCIA**: desplazamiento + aceptación (migración de valor).

Regla maestra:
> El estado se define por comportamiento agregado en una ventana fija, no por velas individuales.

---

## Variables matemáticas mínimas (normalizadas por ATR)
Para cada timestamp `t` (ventana `W=24`):

- `ATR_W = ATR(t-W..t)`
- **Desplazamiento neto**  
  `D = |C_t - C_{t-W}| / ATR_W`
- **Longitud de camino**  
  `L = sum(|C_i - C_{i-1}|) / ATR_W`
- **Eficiencia (Kaufman)**  
  `ER = |C_t - C_{t-W}| / sum(|C_i - C_{i-1}|)` ∈ [0,1]
- **Aceptación `A`**  
  - Rango `R=[L_W, H_W]` de la ventana.
  - Dirección según signo de `C_t - C_{t-W}`.
  - “Zona nueva”: tercio superior (up) o inferior (down).
  - `A = (# cierres en zona nueva durante últimos N=8 H1) / N`.

---

## Auto-etiquetado (y_state) — reglas iniciales
Estas reglas **no se optimizan por accuracy**. Se validan por impacto en gating/PnL.

- **TENDENCIA** si:  
  `ER > 0.35` AND `D > 1.2` AND `A > 0.55`
- **BALANCE** si:  
  `ER < 0.20` AND `D < 0.8` AND `A < 0.50`
- **TRANSICIÓN**: resto de casos (especialmente `D` alto con `A` bajo).

---

## Features del State Engine (máx. 10–12)
**PA-nativas. Sin indicadores clásicos. Sin niveles.**

Core:
1) `D`
2) `ER`
3) `A`
4) `Range_W = (H_W - L_W) / ATR_W`
5) `CloseLocation = (C_t - L_W) / (H_W - L_W)`
6) `ReentryCount` (reingresos al rango tras ruptura)
7) `InsideBarsRatio` (compresión)
8) `SwingCounts` (#HH/#HL o #LL/#LH; fractales simples)

Opcionales (solo si aportan PnL/DD):
9) `EfficiencySlope` (pendiente de ER en W)
10) `RangeSlope` (expansión/contracción)

---

## Modelos
### StateEngine (principal)
- **Modelo**: LightGBM multiclass (BALANCE/TRANSICIÓN/TENDENCIA).
- **Salida**: `P(balance)`, `P(transición)`, `P(tendencia)`.
- **Frecuencia**: cada H1.
- **Calibración**: opcional, solo si mejora gating.

### Modelos auxiliares (raros, con guardrails)
- **AcceptanceModel (binario)**: post-break acceptance vs rejection.  
  Uso: override conservador para forzar TRANSICIÓN si no hay aceptación.
- **VolatilityRegime (binario)**: expansión vs rotación (sin dirección).  
  Uso: priorizar familias de setups.

Máximo **2** auxiliares. Si no mejoran PnL/DD OOS, se eliminan.

---

## Gating (`ALLOW_*`) — política determinista
`ALLOW` **no se entrena**. Es una capa lógica.

Ejemplos:
- `ALLOW_trend_pullback = 1` si `P(tendencia) ≥ 0.60` AND `P(transición) ≤ 0.30`
- `ALLOW_balance_fade = 1` si `P(balance) ≥ 0.60`
- `ALLOW_transition_failure = 1` si `P(transición) ≥ 0.60`

Propósito:
- Prohibir trades inexistentes.
- Reducir fatiga decisional.
- Convertir “no operar” en decisión explícita.

---

## Setups por estado (familias)
- **TENDENCIA**: continuaciones y pullbacks estructurales.
- **BALANCE**: fades de extremos, sweep & reclaim, failed breakouts.
- **TRANSICIÓN**: solo failures/reclaims tácticos.  
  **Prohibido swing direccional.**

El repo no ejecuta entradas automáticas sin confirmación PA.

---

## Evaluación (criterios válidos)
- Confusión crítica: TRANSICIÓN → TENDENCIA (debe ser baja).
- PnL/Trade y Drawdown **condicionados por estado y `ALLOW_*`**.
- Reducción de overtrading vs baseline.
- Walk-forward mensual (out-of-sample).

Éxito mínimo:
- Menos trades con mismo PnL, o
- Igual trades con menor DD, o
- Mayor expectancy aunque baje accuracy.

---

## Anti-leakage
- Features usan solo info ≤ t.
- Labels se calculan offline con horizonte explícito.
- No guardar variables auxiliares usadas para etiquetar como features.
- No usar info futura en detección de setups.

---

## Lista de veto
- Taylor/VWAP, niveles, bandas, confluencias.
- Indicadores clásicos (RSI, MACD, etc.).
- Predicción direccional next-bar.
- RL/policy learning end-to-end.
- Optimizar por F1/accuracy.
- Inflar features sin mejora en PnL/DD.

---

## Resultado esperado
Un sistema que:
- Clasifica estado con alta robustez.
- Habilita/prohíbe familias de setups con reglas claras.
- Reduce errores estructurales.
- Aumenta estabilidad y supervivencia del trader.
- Prioriza claridad y disciplina sobre actividad.
