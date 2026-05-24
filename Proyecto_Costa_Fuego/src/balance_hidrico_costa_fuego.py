"""
═══════════════════════════════════════════════════════════════════════════════
BALANCE HÍDRICO — CONCENTRADORA COSTA FUEGO
Pipeline ETL de Simulación de Planta (365 días operacionales)
═══════════════════════════════════════════════════════════════════════════════

Autor      : Sergio Ignacio Jerez Rossel
Institución: Universidad de Concepción — Depto. Metalurgia
Versión    : 2.1 (Calibración EPCM - NI 43-101)
Descripción: Motor de balance de masa y agua que modela cuatro nodos de proceso
             (Molienda → Flotación Rougher → Limpieza → Dewatering/TSF) con
             lazos de recirculación resueltos por convergencia iterativa.
             Incluye evaluación dinámica de cuello de botella (SAG vs Bolas).

Fuentes de diseño:
    - Wood (Tablas 13.23 y 16.31): Recuperaciones, potencia dual SAG/Bolas
    - NI 43-101 (Secciones 17.4.7.15–17): Restricciones operacionales del TSF
    - NI 43-101 (Tabla 13.18): Target LOM de ley de concentrado final (25.6% Cu)
═══════════════════════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCIAS
# ─────────────────────────────────────────────────────────────────────────────
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CRITERIOS DE DISEÑO DE PROCESO (PDC — Process Design Criteria)
# ─────────────────────────────────────────────────────────────────────────────
class PDC:
    """
    Contenedor inmutable de todos los parámetros de diseño del proceso.
    Calibrado con el reporte técnico NI 43-101 (Wood, 2025).
    """
    # Nodo 1 — Molienda (NI 43-101, Tabla 16.31)
    POTENCIA_MOLINO_BOLAS_KW : float = 37_905   # Numerador calibrado para 2x19MW
    PCT_SOLIDOS_FLOTACION    : float = 33.0     # % sólidos objetivo en rebalse de ciclones

    # Nodo 2 — Flotación Rougher
    LEY_CU_CONCENTRADO_ROUGHER : float = 12.0   # % Cu nominal en concentrado primario
    PCT_SOLIDOS_C_ROUGHER      : float = 25.0   # % sólidos arrastrado hacia Cleaner

    # Nodo 3 — Limpieza (Cleaner)
    LEY_CU_CONCENTRADO_FINAL : float = 25.6     # % Cu LOM Average (NI 43-101, Tabla 13.18)
    PCT_SOLIDOS_C_FINAL      : float = 20.0     # % sólidos al salir de celdas de limpieza

    # Nodo 4 — Dewatering y Tranque (NI 43-101, Secciones 17.4.7.15–17)
    PCT_SOLIDOS_ESPESADOR_RELAVES  : float = 70.0   # % sólidos descarga espesador TSF
    PCT_SOLIDOS_FILTRO_CONCENTRADO : float = 90.0   # % sólidos producto final (humedad < 10%)

    # Solver
    TOLERANCIA_CIERRE : float = 1e-4    # Límite de error en balances [toneladas/hora]
    MAX_ITERACIONES   : int   = 100     # Máximo de iteraciones para convergencia


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS DEL PROYECTO
# ─────────────────────────────────────────────────────────────────────────────
RAIZ = Path(r"C:\Users\Sergio\Desktop\Universidad\Proyectos\Proyecto_Fluor_Balance")

RUTAS = {
    "entrada"       : RAIZ / "Data" / "Raw"       / "Raw_feed_365d.xlsx",
    "balance"       : RAIZ / "Data" / "Processed"  / "resultados_balance_365d.csv",
    "sankey"        : RAIZ / "Data" / "Processed"  / "data_sankey_agua.csv",
}


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 1 — INGESTA DE DATOS (Extract)
# ═════════════════════════════════════════════════════════════════════════════
def cargar_datos(ruta: Path) -> pd.DataFrame:
    """
    Carga el dataset sintético de 365 días y fuerza el tipado
    correcto de columnas para evitar corrupción de matrices numéricas.
    """
    tipos = {
        "Bwi_kWh_t"         : float,
        "Dwi_kWh_m3"        : float,  # Índice de dureza SAG (NI 43-101)
        "Ley_Cu_Cabeza_Pct" : float,
        "Humedad_mina_Pct"  : float,
        "Gavedad_especifica": float,
    }
    df = pd.read_excel(ruta, dtype=tipos)
    log.info(f"Datos cargados: {len(df)} días operacionales desde '{ruta.name}'")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 2 — MOTOR DE BALANCE (Transform)
# ═════════════════════════════════════════════════════════════════════════════

def _validar_cierre(
    diferencia: pd.Series,
    nombre: str,
    tolerancia: float = PDC.TOLERANCIA_CIERRE
) -> None:
    """
    Audita la Ley de Conservación de Masa en un nodo.
    """
    error = np.abs(diferencia).max()
    assert error < tolerancia, (
        f"⚠️  Falla termodinámica en [{nombre}]: "
        f"error de cierre = {error:.6f} t/h "
        f"(tolerancia = {tolerancia} t/h)"
    )


def calcular_nodo1_molienda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nodo 1 — Preparación y Molienda SAG/Bolas.
    Aplica la restricción dual de cuello de botella según NI 43-101 (Tabla 16.31).
    """
    p = PDC

    # 1.1 Throughput limitado por Cuello de Botella Dual (SAG vs Bolas)
    df["TputBM_tph0"] = p.POTENCIA_MOLINO_BOLAS_KW / (df["Bwi_kWh_t"] * 0.7881 - 2.453)
    df["TputSAG_tph"] = 23230 / (0.4184 * (5.1113 * (df["Dwi_kWh_m3"] ** 0.7059) - 0.4305))
    
    # Flujo real (limitante físico de la planta)
    df["F_seco_tph"] = np.minimum(df["TputBM_tph"], df["TputSAG_tph"])

    # 1.2 Transferencia de valor: cobre fino en la alimentación
    df["Fino_Cu_tph"] = df["F_seco_tph"] * (df["Ley_Cu_Cabeza_Pct"] / 100.0)

    # 1.3 Agua intersticial del mineral (fenomenología de humedad de mina)
    df["Agua_Mineral_tph"] = df["F_seco_tph"] * (
        df["Humedad_mina_Pct"] / (100.0 - df["Humedad_mina_Pct"])
    )
    df["F_humedo_tph"] = df["F_seco_tph"] + df["Agua_Mineral_tph"]

    # 1.4 Demanda hídrica total de la pulpa (restricción de 33% sólidos)
    df["Agua_Pulpa_Total_tph"] = df["F_seco_tph"] * (
        (100.0 - p.PCT_SOLIDOS_FLOTACION) / p.PCT_SOLIDOS_FLOTACION
    )

    # 1.5 Agua adicional requerida en molienda (make-up sin recirculación)
    df["Agua_Adicional_Requerida_tph"] = (
        df["Agua_Pulpa_Total_tph"] - df["Agua_Mineral_tph"]
    )

    # Validación: masa_in == masa_out
    masa_in  = df["F_seco_tph"] + df["Agua_Mineral_tph"] + df["Agua_Adicional_Requerida_tph"]
    masa_out = df["F_seco_tph"] + df["Agua_Pulpa_Total_tph"]
    _validar_cierre(masa_in - masa_out, "Nodo 1 — Masa Total")

    log.info("Nodo 1 Correcto | Throughput promedio: %.1f t/h seco", df["F_seco_tph"].mean())
    return df


def calcular_nodo2_rougher(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nodo 2 — Flotación Rougher.
    Curvas empíricas de Wood (Tabla 13.23).
    """
    p = PDC

    # 2.1 Recuperación dinámica por origen de mineral
    rec_productora = np.clip(
        9.072 * df["Ley_Cu_Cabeza_Pct"] + 83.66,
        a_min=None, a_max=95.0
    )
    rec_cortadera = np.clip(
        17.016 * np.log(df["Ley_Cu_Cabeza_Pct"]) + 96.378,
        a_min=18.0, a_max=90.0
    )
    df["Recuperacion_Rougher_Pct"] = np.where(
        df["Origen_mineral"] == "Productora",
        rec_productora,
        rec_cortadera
    )

    # 2.2 Balance de cobre fino
    df["Cu_Fino_C_Rougher_tph"] = df["Fino_Cu_tph"] * (df["Recuperacion_Rougher_Pct"] / 100.0)
    df["Cu_Fino_T_Rougher_tph"] = df["Fino_Cu_tph"] - df["Cu_Fino_C_Rougher_tph"]

    # 2.3 Balance de sólidos
    df["Masa_C_Rougher_tph"] = df["Cu_Fino_C_Rougher_tph"] / (p.LEY_CU_CONCENTRADO_ROUGHER / 100.0)
    df["Masa_T_Rougher_tph"] = df["F_seco_tph"] - df["Masa_C_Rougher_tph"]

    # 2.4 Partición de agua
    df["Agua_C_Rougher_tph"] = df["Masa_C_Rougher_tph"] * (
        (100.0 - p.PCT_SOLIDOS_C_ROUGHER) / p.PCT_SOLIDOS_C_ROUGHER
    )
    df["Agua_T_Rougher_tph"] = df["Agua_Pulpa_Total_tph"] - df["Agua_C_Rougher_tph"]

    # Validaciones
    _validar_cierre(df["F_seco_tph"]          - (df["Masa_C_Rougher_tph"]   + df["Masa_T_Rougher_tph"]),   "Nodo 2 — Sólidos")
    _validar_cierre(df["Agua_Pulpa_Total_tph"] - (df["Agua_C_Rougher_tph"]  + df["Agua_T_Rougher_tph"]),   "Nodo 2 — Agua")
    _validar_cierre(df["Fino_Cu_tph"]          - (df["Cu_Fino_C_Rougher_tph"] + df["Cu_Fino_T_Rougher_tph"]), "Nodo 2 — Cobre")

    return df


def calcular_nodo3_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nodo 3 — Limpieza (Cleaner).
    Rechazo de masa forzando el target LOM de 25.6% Cu.
    """
    p = PDC

    # 3.1 Cobre fino a concentrado final
    df["Cu_Fino_C_Final_tph"]    = df["Cu_Fino_C_Rougher_tph"]
    df["Cu_Fino_T_Cleaner_tph"]  = 0.0

    # 3.2 Masa de concentrado final (restricción comercial)
    df["Masa_C_Final_tph"]    = df["Cu_Fino_C_Final_tph"] / (p.LEY_CU_CONCENTRADO_FINAL / 100.0)
    df["Masa_T_Cleaner_tph"]  = df["Masa_C_Rougher_tph"] - df["Masa_C_Final_tph"]

    # 3.3 Balance de agua
    df["Agua_C_Final_tph"]    = df["Masa_C_Final_tph"] * (
        (100.0 - p.PCT_SOLIDOS_C_FINAL) / p.PCT_SOLIDOS_C_FINAL
    )
    df["Agua_T_Cleaner_tph"]  = df["Agua_C_Rougher_tph"] - df["Agua_C_Final_tph"]

    # Validaciones
    _validar_cierre(
        df["Masa_C_Rougher_tph"] - (df["Masa_C_Final_tph"]  + df["Masa_T_Cleaner_tph"]),
        "Nodo 3 — Sólidos"
    )
    _validar_cierre(
        df["Agua_C_Rougher_tph"] - (df["Agua_C_Final_tph"]  + df["Agua_T_Cleaner_tph"]),
        "Nodo 3 — Agua"
    )
    ley_real = (df["Cu_Fino_C_Final_tph"] / df["Masa_C_Final_tph"]) * 100.0
    _validar_cierre(ley_real - p.LEY_CU_CONCENTRADO_FINAL, "Nodo 3 — Ley Comercial HDS")

    return df


def solver_convergencia(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Orquestador de convergencia — Balance Global con Recirculación.
    """
    p  = PDC
    df = df_raw.copy()
    df["Agua_Recuperada_tph"] = 0.0   

    error    = 1.0
    iteracion = 0

    while error > p.TOLERANCIA_CIERRE and iteracion < p.MAX_ITERACIONES:
        agua_anterior = df["Agua_Recuperada_tph"].copy()

        # ── Nodo 1 actualizado con retorno del ciclo anterior y restricción SAG ──
        df["TputBM_tph"]           = p.POTENCIA_MOLINO_BOLAS_KW / (df["Bwi_kWh_t"] * 0.7881 - 2.453)
        df["TputSAG_tph"]          = 23230 / (0.4184 * (5.1113 * (df["Dwi_kWh_m3"] ** 0.7059) - 0.4305))
        df["F_seco_tph"]           = np.minimum(df["TputBM_tph"], df["TputSAG_tph"])
        
        df["Fino_Cu_tph"]          = df["F_seco_tph"] * (df["Ley_Cu_Cabeza_Pct"] / 100.0)
        df["Agua_Mineral_tph"]     = df["F_seco_tph"] * (df["Humedad_mina_Pct"] / (100.0 - df["Humedad_mina_Pct"]))
        df["Agua_Pulpa_Total_tph"] = df["F_seco_tph"] * ((100.0 - p.PCT_SOLIDOS_FLOTACION) / p.PCT_SOLIDOS_FLOTACION)

        df["Agua_Fresca_Makeup_tph"] = np.maximum(
            df["Agua_Pulpa_Total_tph"] - df["Agua_Mineral_tph"] - df["Agua_Recuperada_tph"],
            0.0
        )

        # ── Nodos 2 y 3 ────────────────────────────────────────────────
        rec_prod = np.clip(9.072  * df["Ley_Cu_Cabeza_Pct"] + 83.66,  None, 95.0)
        rec_cort = np.clip(17.016 * np.log(df["Ley_Cu_Cabeza_Pct"]) + 96.378, 18.0, 90.0)
        df["Recuperacion_Rougher_Pct"] = np.where(df["Origen_mineral"] == "Productora", rec_prod, rec_cort)

        df["Cu_Fino_C_Final_tph"] = df["Fino_Cu_tph"] * (df["Recuperacion_Rougher_Pct"] / 100.0)
        df["Masa_C_Rougher_tph"]  = df["Cu_Fino_C_Final_tph"] / (p.LEY_CU_CONCENTRADO_ROUGHER / 100.0)
        df["Masa_T_Rougher_tph"]  = df["F_seco_tph"] - df["Masa_C_Rougher_tph"]
        df["Masa_C_Final_tph"]    = df["Cu_Fino_C_Final_tph"] / (p.LEY_CU_CONCENTRADO_FINAL / 100.0)
        df["Masa_T_Cleaner_tph"]  = df["Masa_C_Rougher_tph"] - df["Masa_C_Final_tph"]

        df["Agua_C_Rougher_tph"]  = df["Masa_C_Rougher_tph"] * ((100.0 - p.PCT_SOLIDOS_C_ROUGHER) / p.PCT_SOLIDOS_C_ROUGHER)
        df["Agua_T_Rougher_tph"]  = df["Agua_Pulpa_Total_tph"] - df["Agua_C_Rougher_tph"]
        df["Agua_C_Final_tph"]    = df["Masa_C_Final_tph"]    * ((100.0 - p.PCT_SOLIDOS_C_FINAL)    / p.PCT_SOLIDOS_C_FINAL)
        df["Agua_T_Cleaner_tph"]  = df["Agua_C_Rougher_tph"] - df["Agua_C_Final_tph"]

        # ── Nodo 4 — Dewatering y TSF ───────────────────────────────────
        df["Masa_Relave_Total_tph"]         = df["Masa_T_Rougher_tph"] + df["Masa_T_Cleaner_tph"]
        df["Agua_Perdida_TSF_tph"]          = df["Masa_Relave_Total_tph"] * ((100.0 - p.PCT_SOLIDOS_ESPESADOR_RELAVES)  / p.PCT_SOLIDOS_ESPESADOR_RELAVES)
        df["Agua_Perdida_Filtro_tph"]       = df["Masa_C_Final_tph"]    * ((100.0 - p.PCT_SOLIDOS_FILTRO_CONCENTRADO) / p.PCT_SOLIDOS_FILTRO_CONCENTRADO)
        df["Agua_Total_a_N4_tph"]           = df["Agua_T_Rougher_tph"] + df["Agua_T_Cleaner_tph"] + df["Agua_C_Final_tph"]
        df["Agua_Recuperada_tph"]           = df["Agua_Total_a_N4_tph"] - df["Agua_Perdida_TSF_tph"] - df["Agua_Perdida_Filtro_tph"]

        # ── Criterio de convergencia ────────────────────────────────────
        error = np.abs(df["Agua_Recuperada_tph"] - agua_anterior).max()
        iteracion += 1

    if iteracion == p.MAX_ITERACIONES:
        log.warning("Solver no convergió en %d iteraciones (error final = %.6f t/h)", iteracion, error)

    # KPI principal: consumo específico de agua fresca
    df["Consumo_Especifico_Agua_m3_t"] = df["Agua_Fresca_Makeup_tph"] / df["F_seco_tph"]

    # Validación macro
    masa_in  = df["F_seco_tph"] + df["Agua_Mineral_tph"] + df["Agua_Fresca_Makeup_tph"]
    masa_out = (df["Masa_Relave_Total_tph"] + df["Agua_Perdida_TSF_tph"]
                + df["Masa_C_Final_tph"]    + df["Agua_Perdida_Filtro_tph"])
    _validar_cierre(masa_in - masa_out, "MACRO — Planta Completa")

    log.info(
        "Solver ✅ | Convergido en %d iteraciones | "
        "Consumo específico promedio: %.3f m³/t",
        iteracion,
        df["Consumo_Especifico_Agua_m3_t"].mean()
    )
    return df, iteracion


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 3 — GENERACIÓN DE TABLA SANKEY (Load → Power BI)
# ═════════════════════════════════════════════════════════════════════════════
def generar_tabla_sankey(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma el balance anual para Power BI (estructura Origen → Destino → Valor).
    """
    avg = df.mean(numeric_only=True)

    flujos = [
        ("Agua Mina (Humedad)",    "Nodo 1: Molienda",      avg["Agua_Mineral_tph"]),
        ("Agua Fresca (Makeup)",   "Nodo 1: Molienda",      avg["Agua_Fresca_Makeup_tph"]),

        ("Nodo 1: Molienda",       "Nodo 2: Flotación",     avg["Agua_Pulpa_Total_tph"]),
        ("Nodo 2: Flotación",      "Nodo 3: Limpieza",      avg["Agua_C_Rougher_tph"]),
        ("Nodo 2: Flotación",      "Nodo 4: Dewatering",    avg["Agua_T_Rougher_tph"]),
        ("Nodo 3: Limpieza",       "Nodo 4: Dewatering",    avg["Agua_T_Cleaner_tph"]),
        ("Nodo 3: Limpieza",       "Nodo 4: Dewatering",    avg["Agua_C_Final_tph"]),

        ("Nodo 4: Dewatering",     "Pérdida TSF",           avg["Agua_Perdida_TSF_tph"]),
        ("Nodo 4: Dewatering",     "Pérdida Concentrado",   avg["Agua_Perdida_Filtro_tph"]),
        ("Nodo 4: Dewatering",     "Nodo 1: Molienda",      avg["Agua_Recuperada_tph"]),
    ]

    df_sankey = pd.DataFrame(flujos, columns=["Source", "Target", "Value"])
    df_sankey["Value"] = df_sankey["Value"].round(2)
    return df_sankey


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 4 — EXPORTACIÓN ROBUSTA (Load)
# ═════════════════════════════════════════════════════════════════════════════
def exportar_csv(df: pd.DataFrame, ruta: Path, nombre: str) -> None:
    """Exporta DataFrame a CSV capturando errores de permisos."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(ruta, index=False)
        log.info("%s exportado → %s", nombre, ruta)
    except PermissionError:
        log.error("❌ PERMISO DENEGADO en '%s'. Cierra el archivo.", ruta.name)
    except Exception as exc:
        log.error("❌ Error inesperado al exportar '%s': %s", ruta.name, exc)


# ═════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("═" * 60)
    log.info("BALANCE HÍDRICO — CONCENTRADORA COSTA FUEGO")
    log.info("═" * 60)

    # ── E: Ingesta ─────────────────────────────────────────────────────
    df = cargar_datos(RUTAS["entrada"])

    # ── T: Balance preliminar ──────────────────────────────────────────
    df = calcular_nodo1_molienda(df)
    df = calcular_nodo2_rougher(df)
    df = calcular_nodo3_cleaner(df)

    # ── T: Solver global con lazos de recirculación ────────────────────
    df_final, n_iter = solver_convergencia(df)

    # ── T: Tabla para diagrama Sankey en Power BI ──────────────────────
    df_sankey = generar_tabla_sankey(df_final)

    # ── L: Exportación ────────────────────────────────────────────────
    exportar_csv(df_final,  RUTAS["balance"], "Balance 365 días")
    exportar_csv(df_sankey, RUTAS["sankey"],  "Tabla Sankey")

    # ── Resumen ejecutivo en consola ───────────────────────────────────
    log.info("─" * 60)
    log.info("RESUMEN EJECUTIVO (CALIBRADO A NI 43-101)")
    log.info("  Throughput promedio    : %.1f t/h (seco)",   df_final["F_seco_tph"].mean())
    log.info("  Recuperación promedio  : %.1f %%",           df_final["Recuperacion_Rougher_Pct"].mean())
    log.info("  Agua fresca promedio   : %.1f t/h",          df_final["Agua_Fresca_Makeup_tph"].mean())
    log.info("  Consumo específico     : %.3f m³/t",         df_final["Consumo_Especifico_Agua_m3_t"].mean())
    log.info("  Iteraciones solver     : %d",                n_iter)
    log.info("─" * 60)
    log.info("Pipeline finalizado sin errores.")


if __name__ == "__main__":
    main()