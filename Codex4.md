Eres Codex, trabajando en el repo State_Engine_Scorer.

OBJETIVO
Convertir el Event Scorer (M5) en un modelo que realmente capture edge operable.
Hoy AUC_calib ~0.52 (casi azar). Debemos mejorar la señal de entrenamiento, la evaluación y la forma de modelar.
Mantener arquitectura: el watchdog NO opera, NO genera señales; el scorer sólo produce telemetría (edge_score / ranking).

PRINCIPIOS NO NEGOCIABLES
- H1-first gating se mantiene: el scorer solo corre si hay ALLOW_* activo en la última H1 cerrada (ya shift(1)).
- El scorer sigue siendo M5-only para eventos, con contexto H1 merge_asof backward y ctx shift(1).
- Prohibido convertir edge_score en BUY/SELL o automatizar trades. Telemetría solamente.
- Evitar leakage: ninguna feature puede usar información futura, y el label no debe filtrarse a features.
- Complejidad es aceptable si aporta calidad predictiva y evaluación robusta.

TAREA 1 — Reemplazar label binario ruidoso por triple-barrier con retorno continuo
En state_engine/events.py:
1) Modificar label_events() para calcular:
   - entry_price = open(t+1)
   - SL_dist = ATR(t) * sl_mult (ATR de df_m5 con rolling window)
   - TP_dist = reward_r * SL_dist
   - triple barrier (TP/SL/vertical k bars) y output continuo r_outcome:
        * si TP primero => r_outcome = +reward_r
        * si SL primero => r_outcome = -1
        * si no toca nada => mark-to-market al cierre de la barra vertical:
              r_outcome = signed( close_end - entry ) / SL_dist
          (signed según side), con clipping a [-1, +reward_r]
2) Mantener además una etiqueta binaria derivada, pero ahora basada en r_outcome con umbral configurable:
   - label = 1 si r_outcome > r_thr (por defecto r_thr=0.0 o 0.2)
   - label = 0 en caso contrario
3) Resolver el caso “TP y SL en la misma vela” con tie-break mejor que “siempre 0”:
   - implementar tie_break configurable:
        a) "worst" => SL gana
        b) "distance" => aproximar cuál se toca primero comparando distancia desde open de esa vela
   - default: "distance"
4) Guardar en el events_df: entry_price, sl_price, tp_price, label, r_outcome.

TAREA 2 — Mejorar evaluación: ranking (lift@K) por familia y por contexto
En scripts/train_event_scorer.py:
1) Además de AUC, reportar métricas orientadas a ranking:
   - Precision@K y Lift@K (K=10,20,50) sobre el set calib
   - base_rate = mean(label) en calib
   - lift@K = precision@K / base_rate
2) Reportar todo por family_id y global:
   - conteos por family_id (train/calib)
   - base_rate por family_id
   - lift@K por family_id
3) Reportar también por bins de margin_H1 (ej. terciles) para ver si el edge aparece cuando el gating es fuerte.
4) Mantener split temporal (train_ratio actual). NO usar shuffle.

TAREA 3 — Modelado: pasar a “scorer por familia” (bundle) sin romper interfaz
En state_engine/scoring.py (o donde corresponda):
1) Implementar EventScorerBundle:
   - diccionario {family_id: EventScorer}
   - .fit(...) que entrena un modelo por family con sus samples
   - .predict_proba / .predict_score que, dado un dataframe de features y una serie family_id, retorna score por fila usando su modelo correspondiente
2) Mantener compatibilidad hacia afuera:
   - el watchdog debe poder recibir un objeto “scorer” con predict_proba/predict_score.
   - si family_id no existe o no hay modelo para esa familia, retornar NaN o score neutral.
3) Guardado:
   - bundle.save(path, metadata) y load
   - metadata incluye: symbol, feature_count, config de labels (k_bars, reward_r, sl_mult, r_thr, tie_break), fecha train, métricas resumen.

TAREA 4 — Features: asegurar no leakage y agregar features “de familia” sin mirar el futuro
1) Revisar FeatureBuilder:
   - confirmar que rolling windows usen sólo pasado (shift/rolling normal)
   - prohibido usar columnas del label_events (entry/sl/tp/label/r_outcome) como features
2) Mantener add_family_features, pero asegurar que no crea leakage.
3) (Opcional de alto valor) agregar features de “calidad del setup” que sean puramente ex-ante:
   - normalized range / ATR
   - distance to recent high/low in ATR units
   - candle body/wick ratios
   - momentum normalized by ATR
   - time-of-day (server hour) como one-hot o sinusoide
   Todo sin mirar futuro.

TAREA 5 — Ajustes de detección para reducir ruido (sin sobre-optimizar)
En detect_events():
1) BALANCE_FADE:
   - filtrar eventos cuando range_width es demasiado chico: require range_width/atr_short > thr (thr por config, ej 0.8)
2) TREND_PULLBACK:
   - usar momentum normalizado por ATR y exigir |mom| > threshold (ej 0.3) para definir uptrend/downtrend
3) Mantener ALLOW_* gating intacto.

TAREA 6 — Salida clara (UI/telemetría en consola)
1) El script train_event_scorer.py debe imprimir un resumen tipo tabla:
   - total events, labeled, feature_count
   - para cada family_id: samples_train, samples_calib, base_rate_calib, AUC_calib (si aplica), lift@10, lift@20
   - global: AUC_calib, lift@K
2) Guardar además un CSV opcional (models/metrics_{symbol}_event_scorer.csv) con esas métricas.

TAREA 7 — Tests y sanity checks mínimos
1) Agregar tests unitarios o checks ejecutables:
   - label_events no produce leakage (entry is t+1 open; atr indexed at t; future scan starts at t+1)
   - r_outcome está acotado [-1, +reward_r] cuando clip_mtm True
   - tie_break funciona en un caso sintético donde TP y SL se tocan en una vela
2) Agregar un “baseline” simple:
   - score = 0 para todos => lift@K ~ 1.0
   - confirmar que el modelo supere baseline en al menos alguna familia (si no, avisar en logs)

ENTREGABLES
- Código actualizado en:
   - state_engine/events.py
   - state_engine/scoring.py (bundle)
   - scripts/train_event_scorer.py
   - tests/ (o un script de sanity)
- README breve o docstring en train_event_scorer indicando cómo interpretar lift@K y r_outcome.
- No romper imports existentes; mantener compatibilidad con el watchdog.

CRITERIO DE ÉXITO (práctico)
- Que el ranking sea útil: lift@20 > 1.15 en al menos 1–2 familias (ideal) y estable en calibración.
- Que el edge_score tenga interpretación: expected R (o proxy monotónico) para priorizar eventos como telemetría.
- Que no haya leakage.

NOTAS
- Mantener warnings: cambiar floor("H") por floor("h").
- Calibración sklearn: evitar cv='prefit' deprecado; migrar a FrozenEstimator si aplica.
