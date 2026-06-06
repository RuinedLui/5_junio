import csv
import pandas as pd

RUTA_CSV = "encuesta_snacks_mundial_2026_guatemala_2500_respuestas.csv"


# =============================================================================
# FASE 1 — CARGA DE DATOS
# =============================================================================

# --- 1.1 Auditoría con csv (lectura raw) -------------------------------------
# Se usa csv.DictReader para revisar el archivo sin que pandas aplique inferencias de tipo ni convierta vacíos automáticamente.

filas_raw = []
with open(RUTA_CSV, encoding="utf-8-sig") as f:
    lector = csv.DictReader(f)
    columnas = lector.fieldnames
    for fila in lector:
        filas_raw.append(fila)

print(f"Columnas  : {len(columnas)}")
print(f"Registros : {len(filas_raw)}")

# Vacíos por columna
print("\nVacíos por columna:")
for col in columnas:
    vacios = sum(1 for f in filas_raw if not f[col].strip())
    if vacios > 0:
        print(f"  {col}: {vacios}")

# Duplicados por EncuestaID
conteo_ids = {}
for f in filas_raw:
    conteo_ids[f["EncuestaID"]] = conteo_ids.get(f["EncuestaID"], 0) + 1
dups = [k for k, v in conteo_ids.items() if v > 1]
print(f"\nIDs duplicados: {len(dups)}")

# --- 1.2 Carga en pandas -----------------------------------------------------

df = pd.read_csv(RUTA_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
print(f"\nDataFrame cargado — shape: {df.shape}")


# =============================================================================
# FASE 2 — EXPLORACIÓN INICIAL
# =============================================================================

print("\nValores únicos por columna:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()}")

# Distribución de columnas clave para el análisis
print("\nDistribución de columnas clave:")
for col in ["RangoEdad", "Genero", "FrecuenciaConsumoSnacks", "GastoSnacksPartido", "PrecioAdecuado"]:
    print(f"\n── {col}")
    conteo = {}
    for val in df[col]:
        v = val.strip() if val.strip() else "(vacío)"
        conteo[v] = conteo.get(v, 0) + 1
    for k, v in sorted(conteo.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

# Rango de fechas de la encuesta
fechas = sorted(df["FechaEncuesta"].tolist())
print(f"\nRango de fechas: {fechas[0]} → {fechas[-1]}")


# =============================================================================
# FASE 3 — LIMPIEZA DE DATOS
# =============================================================================


df_limpio = df.copy()
n_original = len(df_limpio)

# --- 3.1 Strip general -------------------------------------------------------
# Eliminar espacios en blanco al inicio y final en todas las columnas
for col in df_limpio.columns:
    df_limpio[col] = df_limpio[col].str.strip()

# --- 3.2 Eliminar duplicados -------------------------------------------------
# Se conserva la primera ocurrencia de cada EncuestaID
df_limpio = df_limpio.drop_duplicates(subset="EncuestaID", keep="first")
print(f"Duplicados eliminados : {n_original - len(df_limpio)}")
print(f"Registros restantes   : {len(df_limpio)}")

# --- 3.3 Normalizar capitalización -------------------------------------------
# SaborPreferido tiene inconsistencias: 'QUESO', 'Queso', 'nan' como texto
df_limpio["SaborPreferido"] = df_limpio["SaborPreferido"].replace("nan", "")
df_limpio["SaborPreferido"] = df_limpio["SaborPreferido"].str.title()

# --- 3.4 Imputar vacíos ------------------------------------------------------
# Columnas de texto libre → 'No especificado'
for col in ["Municipio", "Ocupacion", "LugarCompraSnacks", "SeleccionInfluyeCompra"]:
    df_limpio[col] = df_limpio[col].replace("", "No especificado")

# Columnas categóricas con pocos vacíos → moda
for col in ["SaborPreferido", "PrecioAdecuado"]:
    moda = df_limpio[df_limpio[col] != ""][col].value_counts().idxmax()
    df_limpio[col] = df_limpio[col].replace("", moda)
    print(f"  {col}: vacíos imputados con moda '{moda}'")

# --- 3.5 Convertir tipos de datos --------------------------------------------
df_limpio["EncuestaID"]    = df_limpio["EncuestaID"].astype(int)
df_limpio["FechaEncuesta"] = pd.to_datetime(df_limpio["FechaEncuesta"], format="%Y-%m-%d", errors="coerce")
df_limpio["HoraEncuesta"]  = pd.to_datetime(df_limpio["HoraEncuesta"], format="%H:%M:%S", errors="coerce").dt.time

print(f"\nRegistros limpios finales: {len(df_limpio)}")


# =============================================================================
# FASE 4 — TRANSFORMACIÓN
# =============================================================================

print("\n" + "=" * 60)
print("FASE 4 — TRANSFORMACIÓN")
print("=" * 60)

# --- 4.1 Segmento de edad ----------------------------------------------------
# Mapeo de RangoEdad a segmentos de mercado para análisis demográfico.
mapa_segmento = {
    "Menos de 18 años": "Juvenil",
    "18 - 24 años"    : "Joven Adulto",
    "25 - 34 años"    : "Adulto Joven",
    "35 - 44 años"    : "Adulto",
    "45 - 54 años"    : "Adulto Mayor",
    "55 años o más"   : "Senior"
}
df_limpio["SegmentoEdad"] = df_limpio["RangoEdad"].map(mapa_segmento).fillna("No especificado")

# --- 4.2 Categoría de precio -------------------------------------------------
# Clasifica el precio unitario aceptado en rangos de mercado.
mapa_precio = {
    "Menos de Q10": "Economico",
    "Q10 - Q15"   : "Accesible",
    "Q16 - Q20"   : "Moderado",
    "Q21 - Q30"   : "Premium",
    "Más de Q30"  : "Super Premium"
}
df_limpio["CategoriaPrecio"] = df_limpio["PrecioAdecuado"].map(mapa_precio).fillna("No especificado")

# --- 4.3 Cantidad de snacks por encuestado -----------------------------------
# SnacksSeleccionados es multi-valor separado por ';', se cuenta cuántos eligió.
df_limpio["CantidadSnacks"] = df_limpio["SnacksSeleccionados"].apply(
    lambda x: len([s for s in x.split(";") if s.strip()])
)

# --- 4.4 Indicador de alta intención de compra -------------------------------
# 1 si el encuestado pagaría más por edición Mundial, 0 si no.
df_limpio["AltaIntencionCompra"] = df_limpio["PagaMasEdicionMundial"].apply(
    lambda x: 1 if x.lower() == "sí" else 0
)

# --- 4.5 Variables temporales ------------------------------------------------
df_limpio["MesEncuesta"] = df_limpio["FechaEncuesta"].dt.month

print("Columnas derivadas creadas:")
for col in ["SegmentoEdad", "CategoriaPrecio", "CantidadSnacks", "AltaIntencionCompra", "MesEncuesta"]:
    print(f"  + {col}")


# =============================================================================
# FASE 5 — STAR SCHEMA
# =============================================================================
# Estructura:
#   fact_encuesta (tabla central de hechos)
#       ├── dim_encuestado
#       ├── dim_ubicacion
#       ├── dim_tiempo
#       ├── dim_snack     
#       └── dim_campania
# =============================================================================


# --- dim_encuestado ----------------------------------------------------------
# Características demográficas del encuestado.
dim_encuestado = df_limpio[["EncuestaID", "RangoEdad", "SegmentoEdad", "Genero", "Ocupacion"]].copy()
dim_encuestado = dim_encuestado.rename(columns={"EncuestaID": "id_encuestado"})
dim_encuestado = dim_encuestado.drop_duplicates(subset="id_encuestado").reset_index(drop=True)
print(f"dim_encuestado : {dim_encuestado.shape}")

# --- dim_ubicacion -----------------------------------------------------------
# Ubicación geográfica del encuestado.
dim_ubicacion = df_limpio[["EncuestaID", "Departamento", "Municipio"]].copy()
dim_ubicacion = dim_ubicacion.rename(columns={"EncuestaID": "id_ubicacion"})
dim_ubicacion = dim_ubicacion.drop_duplicates(subset="id_ubicacion").reset_index(drop=True)
print(f"dim_ubicacion  : {dim_ubicacion.shape}")

# --- dim_tiempo --------------------------------------------------------------
# Información temporal de la encuesta.
dim_tiempo = df_limpio[["EncuestaID", "FechaEncuesta", "MesEncuesta", "HoraEncuesta"]].copy()
dim_tiempo = dim_tiempo.rename(columns={"EncuestaID": "id_tiempo"})
dim_tiempo = dim_tiempo.drop_duplicates(subset="id_tiempo").reset_index(drop=True)
print(f"dim_tiempo     : {dim_tiempo.shape}")

# --- dim_snack ---------------------------------------------------------------
# Un snack por fila 
# Se agrega id_encuestado para poder relacionarla con la fact table.
dim_snack = df_limpio[["EncuestaID", "SnacksSeleccionados"]].copy()
dim_snack["Snack"] = dim_snack["SnacksSeleccionados"].str.split(";")
dim_snack = dim_snack.explode("Snack")
dim_snack["Snack"] = dim_snack["Snack"].str.strip()
dim_snack = dim_snack[dim_snack["Snack"] != ""].reset_index(drop=True)
dim_snack = dim_snack.drop(columns=["SnacksSeleccionados"])
dim_snack = dim_snack.rename(columns={"EncuestaID": "id_encuestado"})
print(f"dim_snack      : {dim_snack.shape}")

# --- dim_campania ------------------------------------------------------------
# Preferencias publicitarias y de campaña del encuestado.
dim_campania = df_limpio[[
    "EncuestaID", "TipoPublicidadAtractiva", "PromocionPreferida",
    "CampaniaMasProbableCompra", "CompraTarjetasColeccionables"
]].copy()
dim_campania = dim_campania.rename(columns={"EncuestaID": "id_campania"})
dim_campania = dim_campania.drop_duplicates(subset="id_campania").reset_index(drop=True)
print(f"dim_campania   : {dim_campania.shape}")

# --- fact_encuesta ------------------------------------------------------------
# Tabla central con métricas y claves foráneas hacia las dimensiones.
fact_encuesta = df_limpio[[
    "EncuestaID",
    "GastoSnacksPartido",
    "CantidadSnacks",
    "AltaIntencionCompra",
    "CategoriaPrecio",
    "FrecuenciaConsumoSnacks",
    "SeleccionApoya",
    "JugadoresInfluyentes",
    "PlaneaVerMundial2026",
    "ConQuienVePartidos"
]].copy()

# Claves foráneas (apuntan a EncuestaID en cada dimensión)
fact_encuesta = fact_encuesta.rename(columns={
    "EncuestaID": "id_encuesta"
})
fact_encuesta["id_encuestado"] = fact_encuesta["id_encuesta"]
fact_encuesta["id_ubicacion"]  = fact_encuesta["id_encuesta"]
fact_encuesta["id_tiempo"]     = fact_encuesta["id_encuesta"]
fact_encuesta["id_campania"]   = fact_encuesta["id_encuesta"]

print(f"fact_encuesta  : {fact_encuesta.shape}")

# --- Exportar star schema a CSV ----------------------------------------------
dim_encuestado.to_csv("dim_encuestado.csv",  index=False, encoding="utf-8-sig")
dim_ubicacion.to_csv("dim_ubicacion.csv",    index=False, encoding="utf-8-sig")
dim_tiempo.to_csv("dim_tiempo.csv",          index=False, encoding="utf-8-sig")
dim_snack.to_csv("dim_snack.csv",            index=False, encoding="utf-8-sig")
dim_campania.to_csv("dim_campania.csv",      index=False, encoding="utf-8-sig")
fact_encuesta.to_csv("fact_encuesta.csv",    index=False, encoding="utf-8-sig")
df_limpio.to_csv("encuesta_limpia.csv",      index=False, encoding="utf-8-sig")

print("\nArchivos exportados exitosamente:")
for nombre in ["dim_encuestado", "dim_ubicacion", "dim_tiempo", "dim_snack", "dim_campania", "fact_encuesta", "encuesta_limpia"]:
    print(f"  {nombre}.csv")
 # =============================================================================
# FASE 6 — GENERACIÓN DE REPORTES (Reportes 1 al 3)
# =============================================================================

print("\n" + "=" * 60)
print("FASE 6 — REPORTES EMPRESARIALES")
print("=" * 60)

# -----------------------------------------------------------------------------
# REPORTE 1: PERFIL DEMOGRÁFICO
# Objetivo: Entender quién es nuestro público objetivo (Edad y Género).
# -----------------------------------------------------------------------------
# =============================================================================
# FASE 6 — GENERACIÓN DE REPORTES (Reportes 1 al 3)
# =============================================================================

print("\n" + "=" * 60)
print("FASE 6 — REPORTES EMPRESARIALES")
print("=" * 60)

# -----------------------------------------------------------------------------
# REPORTE 1: PERFIL DEMOGRÁFICO
# Objetivo: Entender quién es nuestro público objetivo (Edad y Género).
# -----------------------------------------------------------------------------
print("\n--- REPORTE 1: PERFIL DEMOGRÁFICO ---")

# Calculamos la distribución por Género y Segmento de Edad mostrando porcentajes
reporte_demografico = pd.crosstab(
    df_limpio['SegmentoEdad'], 
    df_limpio['Genero'], 
    normalize='all' # Esto nos da el porcentaje del total
) * 100

# Redondeamos a 2 decimales para que sea legible
reporte_demografico = reporte_demografico.round(2)
# Añadimos una columna de "Total" por fila
reporte_demografico['Total Segmento (%)'] = reporte_demografico.sum(axis=1)

print("Distribución Porcentual por Edad y Género:\n")
print(reporte_demografico.sort_values(by='Total Segmento (%)', ascending=False))


# -----------------------------------------------------------------------------
# REPORTE 2: FRECUENCIA DE CONSUMO
# Objetivo: Identificar los hábitos de compra del público.
# -----------------------------------------------------------------------------
print("\n--- REPORTE 2: FRECUENCIA DE CONSUMO ---")

# Contamos cuántas personas consumen en cada frecuencia y sacamos el porcentaje
reporte_frecuencia = df_limpio['FrecuenciaConsumoSnacks'].value_counts().reset_index()
reporte_frecuencia.columns = ['Frecuencia', 'Cantidad de Clientes']
reporte_frecuencia['Porcentaje (%)'] = round((reporte_frecuencia['Cantidad de Clientes'] / len(df_limpio)) * 100, 2)

print("Frecuencia de Consumo General:\n")
print(reporte_frecuencia)

# Extra: Cruzar frecuencia con Alta Intención de Compra (muy valioso para el negocio)
frecuencia_vs_intencion = df_limpio.groupby('FrecuenciaConsumoSnacks')['AltaIntencionCompra'].mean().reset_index()
frecuencia_vs_intencion['AltaIntencionCompra'] = round(frecuencia_vs_intencion['AltaIntencionCompra'] * 100, 2)
frecuencia_vs_intencion.columns = ['Frecuencia', 'Probabilidad de Pagar Más (%)']

print("\nProbabilidad de pagar más por la edición Mundial según frecuencia de consumo:\n")
print(frecuencia_vs_intencion.sort_values(by='Probabilidad de Pagar Más (%)', ascending=False))


# -----------------------------------------------------------------------------
# REPORTE 3: RANKING DE SNACKS Y SABORES
# Objetivo: Descubrir los productos estrella para la campaña.
# -----------------------------------------------------------------------------
print("\n--- REPORTE 3: RANKING DE SNACKS ---")

# 1. Ranking de TIPOS de snacks (Usamos dim_snack que ya explotaste con explode() en la Fase 5)
ranking_snacks = dim_snack['Snack'].value_counts().reset_index()
ranking_snacks.columns = ['Tipo de Snack', 'Menciones']
ranking_snacks['Porcentaje de Preferencia (%)'] = round((ranking_snacks['Menciones'] / len(df_limpio)) * 100, 2)

print("Top 5 - Snacks Más Populares:\n")
print(ranking_snacks.head(5))

# 2. Ranking de SABORES (Usamos el df_limpio)
ranking_sabores = df_limpio['SaborPreferido'].value_counts().reset_index()
ranking_sabores.columns = ['Sabor', 'Preferencias']
ranking_sabores['Porcentaje (%)'] = round((ranking_sabores['Preferencias'] / len(df_limpio)) * 100, 2)

print("\nTop 5 - Sabores Más Populares:\n")
print(ranking_sabores.head(5))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# EXPORTACIÓN DE LOS REPORTES 1 al 3
# -----------------------------------------------------------------------------
#reporte_demografico.to_csv("reporte_1_demografico.csv", encoding="utf-8-sig")
#reporte_frecuencia.to_csv("reporte_2_frecuencia.csv", index=False, encoding="utf-8-sig")
#ranking_snacks.to_csv("reporte_3_ranking_snacks.csv", index=False, encoding="utf-8-sig")
#print("\n¡Reportes 1, 2 y 3 exportados exitosamente!")