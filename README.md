# Contención de incendios forestales con DQN y Double DQN (SimFire)

Trabajo final de **Aprendizaje por Refuerzo I** - Especialización en Inteligencia Artificial (UBA).
Desafío Nº 1.

Un agente decide, turno a turno, en qué celda de una grilla colocar un cortafuego mientras el fuego se propaga según el viento y el combustible. El objetivo es **minimizar el área quemada**. Se entrenan y comparan DQN, Double DQN y dos variantes de la recompensa sobre el simulador [SimFire](https://github.com/mitrefireline/simfire) (proyecto Fireline de MITRE).

> **Abrir en Colab:** `https://colab.research.google.com/github/TU_USUARIO/TU_REPOSITORIO/blob/main/incendios_dqn.ipynb`
> _(reemplazar `TU_USUARIO` y `TU_REPOSITORIO` por los del repositorio)_

## Resultados

Evaluación final en 50 escenarios no vistos (media ± desvío entre escenarios; 3 semillas de entrenamiento por agente). Área quemada sobre un máximo de 256 celdas, menos es mejor.

| Política | Recompensa | Área quemada |
|---|---|---|
| Sin acción | -21.10 ± 8.20 | 212.0 ± 82.0 |
| Azar | -14.14 ± 5.43 | 142.4 ± 54.3 |
| Heurística perímetro | -6.51 ± 3.75 | 66.1 ± 37.5 |
| DQN (con shaping) | -7.52 ± 4.74 | 76.2 ± 47.4 |
| Double DQN (con shaping) | -7.39 ± 4.45 | 74.9 ± 44.5 |
| DQN sin shaping | -12.17 ± 5.13 | 122.7 ± 51.3 |
| DQN bonus decreciente | -7.32 ± 4.43 | 74.2 ± 44.3 |

Qué se puede y qué no se puede concluir:

- Con *reward shaping*, DQN y Double DQN reducen el área quemada de ~142 (azar) a ~75 y quedan **cerca de la heurística** (66), que no superan.
- **No hay diferencia demostrable entre DQN y Double DQN**: la diferencia (1.3 celdas) es mucho menor que el desvío entre escenarios (~45).
- Sin shaping el agente aprende, pero poco (~123).
- El **bonus decreciente es un resultado negativo**: su cifra final (74.2) se parece a la de DQN porque el mejor checkpoint de cada semilla es anterior a la bajada del bonus, pero en la curva de validación el área sube hasta ~100 al final del entrenamiento.
- Los desvíos son entre escenarios, no el error de la media.

## El problema

| | |
|---|---|
| **Simulador** | SimFire (modelo de propagación de Rothermel) |
| **Mapa** | Grilla de 16x16 (256 acciones), `pixel_scale` 300, humedad 0.08, viento de 8 mph |
| **Escenarios** | Cada episodio cambia el punto de ignición (zona central) y la dirección del viento (8 direcciones) |
| **Estado** | Imagen de 9 canales de 16x16: estado de la celda (4 canales), combustible, elevación y viento (3 canales) |
| **Acción** | Celda donde se coloca el cortafuego (se enmascaran las celdas quemadas o con cortafuego) |
| **Recompensa** | -0.1 por cada celda que se prende fuego en el turno |

## Método

- **Agentes:** DQN y Double DQN con replay buffer, red target y epsilon-greedy con máscara de acciones inválidas.
- **Red:** totalmente convolucional con arquitectura *dueling*.
- **Hiperparámetros:** learning rate 5e-5 (Adam), gamma 0.9, batch 64, buffer 30000, target cada 1000 pasos, epsilon de 1.0 a 0.05 en 40000 pasos, 1500 episodios.
- **Reward shaping:** bonus de +0.3 si el cortafuego se coloca pegado a una celda en llamas. Las curvas y la evaluación miden siempre solo el área quemada, sin bonus.
- **Validación:** cada 100 episodios se evalúa sin azar en 40 escenarios y se conserva el mejor checkpoint. La evaluación final usa 50 escenarios distintos.
- **Baselines:** sin acción, azar y una heurística de "perímetro" (cortafuego al azar entre las celdas libres pegadas a una celda en llamas; es un baseline propio, no una técnica de la literatura).

## Cómo ejecutarlo

### En Google Colab (recomendado)

1. Abrí `incendios_dqn.ipynb` en Colab (botón de arriba, o subiendo el archivo).
2. Activá GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4*.
3. **Prueba rápida primero:** poné `MODO_RAPIDO = True` en la celda de parámetros y ejecutá todo (~2 minutos). Sirve para confirmar que la instalación de SimFire funciona.
4. Corrida completa: `MODO_RAPIDO = False` y *Entorno de ejecución → Ejecutar todo*.

La primera celda instala SimFire sola. Una corrida completa (4 agentes × 3 semillas × 1500 episodios) tardó entre 1.5 y 2 horas en nuestra ejecución (según los tiempos de los logs, cada entrenamiento de 1500 episodios llevó entre 6 y 10 minutos); con una sola semilla (`SEMILLAS_ENTRENAMIENTO = [0]`) son unos 30 minutos.

> **Importante:** el disco de Colab es temporal. Si la sesión se cae se pierden los resultados. Poné `GUARDAR_EN_DRIVE = True` para guardar en tu Google Drive (Colab te pide autorización), o descargá la carpeta `resultados/` antes de cerrar.

### Parámetros útiles (celda "2. Parámetros")

| Parámetro | Qué hace |
|---|---|
| `MODO_RAPIDO` | Corrida corta para probar que todo funciona |
| `SEMILLAS_ENTRENAMIENTO` | Semillas por agente (por defecto `[0, 1, 2]`) |
| `SOLO_AGENTES` | Entrena solo los agentes que indiques, p. ej. `["DQN bonus decreciente"]`. Guarda en `resultados_parcial` |
| `BONUS_FRONTERA` | Valor del reward shaping |
| `GUARDAR_EN_DRIVE` | Guarda los resultados en Google Drive |
| `CARGAR_AGENTES_GUARDADOS` | Reconstruye los agentes desde `pesos_*.pt` sin volver a entrenar |

### Como script

`incendios_dqn.py` es el mismo código en formato script (con marcadores `# %%` por celda, compatible con Jupytext). También instala SimFire sola.

## Archivos

| Archivo | Contenido |
|---|---|
| `incendios_dqn.ipynb` | Notebook para Colab (sin salidas guardadas) |
| `incendios_dqn.py` | El mismo código en formato script |

Al correr se genera la carpeta `resultados/` con las curvas de convergencia, la comparativa de agentes, la validación durante el entrenamiento, la evaluación final, los mapas de cada política, `resumen.json`, `curvas_entrenamiento.npz` y los pesos de cada agente (`pesos_*.pt`).

## Notas técnicas

- **SimFire no instala con `pip install simfire` en Colab:** el paquete declara `python ~=3.9` y Colab usa 3.12. El notebook lo instala con `--no-deps --ignore-requires-python` y las dependencias a mano.
- **`landfire`, `geotiff`, `geopandas` y `geopy`** (solo sirven para mapas reales) no se pueden instalar en Python 3.12 y rompen el import de SimFire. El notebook los reemplaza por módulos vacíos antes de importarlo, porque se usa terreno generado por funciones.
- Se usa `FireSimulation` de SimFire directamente. No se usa SimHarness (depende de Ray/RLlib y de Python 3.9).
- SimFire dibuja con matplotlib al crear un escenario; por eso en la sección de mapas se juegan primero todos los episodios y recién después se dibuja.

## Limitaciones

- Grilla chica (16x16), terreno fijo, una sola acción (cortafuego) y 3 semillas por agente.
- El agente con shaping aprende a imitar el criterio de la heurística, que es justamente lo que premia el bonus; no se demostró que la supere.
- La instalación de SimFire en Python 3.12 se verificó en un entorno local de 3.12 y en la ejecución de Colab del autor; puede variar con futuras versiones de Colab.

## Referencias

- [SimFire / Fireline (MITRE)](https://github.com/mitrefireline/simfire)
- Mnih et al., *Human-level control through deep reinforcement learning* (DQN)
- van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning* (Double DQN)
- Wang et al., *Dueling Network Architectures for Deep Reinforcement Learning*
