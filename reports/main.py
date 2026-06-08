import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RUTA_CSV = "../data/encuesta_snacks_mundial_2026_guatemala_2500_respuestas.csv"


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
dim_encuestado.to_csv("../data/dim_encuestado.csv",  index=False, encoding="utf-8-sig")
dim_ubicacion.to_csv("../data/dim_ubicacion.csv",    index=False, encoding="utf-8-sig")
dim_tiempo.to_csv("../data/dim_tiempo.csv",          index=False, encoding="utf-8-sig")
dim_snack.to_csv("../data/dim_snack.csv",            index=False, encoding="utf-8-sig")
dim_campania.to_csv("../data/dim_campania.csv",      index=False, encoding="utf-8-sig")
fact_encuesta.to_csv("../data/fact_encuesta.csv",    index=False, encoding="utf-8-sig")
df_limpio.to_csv("../data/encuesta_limpia.csv",      index=False, encoding="utf-8-sig")

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
# REPORTE 4: ANÁLISIS DE PRECIOS
# Objetivo: Determinar el rango de precio ideal para el lanzamiento.
# -----------------------------------------------------------------------------
print("\n--- REPORTE 4: ANÁLISIS DE PRECIOS ---")

reporte_precios = df_limpio['PrecioAdecuado'].value_counts().reset_index()
reporte_precios.columns = ['Rango de Precio', 'Cantidad']

reporte_precios['Porcentaje (%)'] = round(
    (reporte_precios['Cantidad'] / len(df_limpio)) * 100,
    2
)

print("Preferencia de precios:\n")
print(reporte_precios)

# -----------------------------------------------------------------------------
# REPORTE 5: EQUIPOS CON MAYOR INFLUENCIA
# Objetivo: Identificar qué selecciones generan mayor impacto comercial.
# -----------------------------------------------------------------------------
print("\n--- REPORTE 5: EQUIPOS CON MAYOR INFLUENCIA ---")

ranking_selecciones = (
    df_limpio['SeleccionInfluyeCompra']
    .value_counts()
    .reset_index()
)

ranking_selecciones.columns = ['Selección', 'Menciones']

ranking_selecciones['Porcentaje (%)'] = round(
    (ranking_selecciones['Menciones'] / len(df_limpio)) * 100,
    2
)

print("Top 5 - Selecciones con mayor influencia:\n")
print(ranking_selecciones.head(5))


# -----------------------------------------------------------------------------
# REPORTE 6: JUGADORES CON MAYOR INFLUENCIA 
# Objetivo: Identificar qué jugadores son los más mencionados por los encuestados.
# -----------------------------------------------------------------------------
print("\n--- REPORTE 6: JUGADORES CON MAYOR INFLUENCIA ---")

# Dado que JugadoresInfluyentes viene separado por ';', usamos explode para contarlos individualmente
jugadores_lista = df_limpio['JugadoresInfluyentes'].str.split(';').explode().str.strip()
ranking_jugadores = jugadores_lista[jugadores_lista != ""].value_counts().reset_index()
ranking_jugadores.columns = ['Jugador', 'Menciones']
ranking_jugadores['Porcentaje (%)'] = round((ranking_jugadores['Menciones'] / len(df_limpio)) * 100, 2)

print("Top 5 - Jugadores con mayor influencia:\n")
print(ranking_jugadores.head(5))


# -----------------------------------------------------------------------------
# REPORTE 7: TIPO DE PUBLICIDAD MÁS EFECTIVA 
# Objetivo: Determinar qué medios publicitarios tienen mayor alcance/impacto.
# -----------------------------------------------------------------------------
print("\n--- REPORTE 7: TIPO DE PUBLICIDAD MÁS EFECTIVA ---")

ranking_publicidad = df_limpio['TipoPublicidadAtractiva'].value_counts().reset_index()
ranking_publicidad.columns = ['Tipo Publicidad', 'Cantidad']
ranking_publicidad['Porcentaje (%)'] = round((ranking_publicidad['Cantidad'] / len(df_limpio)) * 100, 2)

print(ranking_publicidad)


# -----------------------------------------------------------------------------
# REPORTE 8: PROMOCIONES PREFERIDAS 
# Objetivo: Conocer qué tipo de ofertas incentivan más la compra.
# -----------------------------------------------------------------------------
print("\n--- REPORTE 8: PROMOCIONES PREFERIDAS ---")

ranking_promociones = df_limpio['PromocionPreferida'].value_counts().reset_index()
ranking_promociones.columns = ['Promoción', 'Cantidad']
ranking_promociones['Porcentaje (%)'] = round((ranking_promociones['Cantidad'] / len(df_limpio)) * 100, 2)

print(ranking_promociones)

# -----------------------------------------------------------------------------
# REPORTE 9: INTENCIÓN DE COMPRA POR CAMPAÑA
# Objetivo: Identificar qué edición/campaña tiene mayor probabilidad de generar compra.
# -----------------------------------------------------------------------------
print("\n--- REPORTE 9: INTENCIÓN DE COMPRA POR CAMPAÑA ---")

# Distribución general de la campaña más probable de compra
ranking_campania = df_limpio['CampaniaMasProbableCompra'].value_counts().reset_index()
ranking_campania.columns = ['Campaña', 'Cantidad']
ranking_campania['Porcentaje (%)'] = round((ranking_campania['Cantidad'] / len(df_limpio)) * 100, 2)

print("Campañas con mayor intención de compra:\n")
print(ranking_campania)

# Cruce: campaña preferida vs alta intención de pagar más (AltaIntencionCompra)
campania_vs_intencion = df_limpio.groupby('CampaniaMasProbableCompra')['AltaIntencionCompra'].mean().reset_index()
campania_vs_intencion['AltaIntencionCompra'] = round(campania_vs_intencion['AltaIntencionCompra'] * 100, 2)
campania_vs_intencion.columns = ['Campaña', 'Clientes Dispuestos a Pagar Más (%)']

print("\nCampañas con mayor conversión a pago premium:\n")
print(campania_vs_intencion.sort_values(by='Clientes Dispuestos a Pagar Más (%)', ascending=False))


# -----------------------------------------------------------------------------
# REPORTE 10: RECOMENDACIÓN ESTRATÉGICA FINAL
# Objetivo: Consolidar los hallazgos clave en una propuesta de campaña basada en datos.
# -----------------------------------------------------------------------------
print("\n--- REPORTE 10: RECOMENDACIÓN ESTRATÉGICA FINAL ---")

# Snack más popular
snack_top = dim_snack['Snack'].value_counts().idxmax()

# Sabor más preferido
sabor_top = df_limpio['SaborPreferido'].value_counts().idxmax()

# Segmento de edad dominante
segmento_top = df_limpio['SegmentoEdad'].value_counts().idxmax()

# Selección con mayor influencia en compra (normalizar capitalización primero)
seleccion_top = df_limpio['SeleccionInfluyeCompra'].str.title().value_counts().idxmax()

# Jugador más influyente
jugador_top = df_limpio['JugadoresInfluyentes'].str.split(';').explode().str.strip().value_counts().idxmax()

# Tipo de publicidad más efectiva
publicidad_top = df_limpio['TipoPublicidadAtractiva'].value_counts().idxmax()

# Promoción preferida
promocion_top = df_limpio['PromocionPreferida'].value_counts().idxmax()

# Precio ideal (moda)
precio_top = df_limpio['PrecioAdecuado'].value_counts().idxmax()

# Campaña con mayor intención de compra
campania_top = df_limpio['CampaniaMasProbableCompra'].value_counts().idxmax()

print("PROPUESTA DE CAMPAÑA — SNACKS × MUNDIAL FIFA 2026")
print("=" * 60)
print(f"  Snack a promocionar    : {snack_top}")
print(f"  Sabor principal        : {sabor_top}")
print(f"  Segmento objetivo      : {segmento_top}")
print(f"  Selección influyente   : {seleccion_top}")
print(f"  Jugador embajador      : {jugador_top}")
print(f"  Canal publicitario     : {publicidad_top}")
print(f"  Promoción recomendada  : {promocion_top}")
print(f"  Precio ideal           : {precio_top}")
print(f"  Campaña de empaque     : {campania_top}")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# EXPORTACIÓN DE LOS REPORTES 1 al 3
# -----------------------------------------------------------------------------
#reporte_demografico.to_csv("reporte_1_demografico.csv", encoding="utf-8-sig")
#reporte_frecuencia.to_csv("reporte_2_frecuencia.csv", index=False, encoding="utf-8-sig")
#ranking_snacks.to_csv("reporte_3_ranking_snacks.csv", index=False, encoding="utf-8-sig")

# ranking_jugadores.to_csv("reporte_4_jugadores.csv", index=False, encoding="utf-8-sig")
# ranking_publicidad.to_csv("reporte_5_publicidad.csv", index=False, encoding="utf-8-sig")
# ranking_promociones.to_csv("reporte_6_promociones.csv", index=False, encoding="utf-8-sig")

#print("\n¡Reportes 1, 2 y 3 exportados exitosamente!")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Dashboard Ejecutivo — Snacks × Mundial FIFA 2026 | Guatemala",
             fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

# ── Gráfico 1: Top 5 Snacks ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
top_snacks = dim_snack["Snack"].value_counts().head(5)
ax1.barh(top_snacks.index[::-1], top_snacks.values[::-1], color="#2196F3")
ax1.set_title("Top 5 Snacks", fontweight="bold")
ax1.set_xlabel("Menciones")

# ── Gráfico 2: Distribución por Segmento de Edad ─────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
segmentos = df_limpio["SegmentoEdad"].value_counts()
ax2.pie(segmentos.values, labels=segmentos.index, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 8})
ax2.set_title("Segmento de Edad", fontweight="bold")

# ── Gráfico 3: Distribución por Género ───────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
generos = df_limpio["Genero"].value_counts()
ax3.bar(generos.index, generos.values, color=["#E91E63", "#2196F3", "#9E9E9E"])
ax3.set_title("Distribución por Género", fontweight="bold")
ax3.set_ylabel("Cantidad")

# ── Gráfico 4: Top 5 Jugadores Influyentes ────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
top_jugadores = df_limpio["JugadoresInfluyentes"].str.split(";").explode().str.strip()
top_jugadores = top_jugadores[top_jugadores != ""].value_counts().head(5)
ax4.barh(top_jugadores.index[::-1], top_jugadores.values[::-1], color="#FF9800")
ax4.set_title("Top 5 Jugadores Influyentes", fontweight="bold")
ax4.set_xlabel("Menciones")

# ── Gráfico 5: Tipo de Publicidad más Efectiva ────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
publicidad = df_limpio["TipoPublicidadAtractiva"].value_counts()
ax5.barh(publicidad.index[::-1], publicidad.values[::-1], color="#4CAF50")
ax5.set_title("Publicidad más Efectiva", fontweight="bold")
ax5.set_xlabel("Cantidad")

# ── Gráfico 6: Promociones Preferidas ────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
promociones = df_limpio["PromocionPreferida"].value_counts()
ax6.barh(promociones.index[::-1], promociones.values[::-1], color="#9C27B0")
ax6.set_title("Promociones Preferidas", fontweight="bold")
ax6.set_xlabel("Cantidad")

# ── Gráfico 7: Intención de Compra por Campaña ───────────────────────────────
ax7 = fig.add_subplot(gs[2, 0:2])
campanas = df_limpio["CampaniaMasProbableCompra"].value_counts()
ax7.bar(campanas.index, campanas.values, color="#F44336")
ax7.set_title("Intención de Compra por Campaña", fontweight="bold")
ax7.set_ylabel("Cantidad")
plt.setp(ax7.get_xticklabels(), rotation=20, ha="right", fontsize=8)

# ── KPIs texto ────────────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis("off")
kpis = (
    f"PROPUESTA DE CAMPAÑA\n"
    f"{'─' * 28}\n"
    f"Snack       : {snack_top}\n"
    f"Sabor       : {sabor_top}\n"
    f"Segmento    : {segmento_top}\n"
    f"Selección   : {seleccion_top}\n"
    f"Jugador     : {jugador_top}\n"
    f"Publicidad  : {publicidad_top}\n"
    f"Promoción   : {promocion_top}\n"
    f"Precio      : {precio_top}\n"
    f"Campaña     : {campania_top}"
)
ax8.text(0.05, 0.95, kpis, transform=ax8.transAxes, fontsize=9,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))

plt.savefig("../images/dashboard_ejecutivo.png", dpi=150, bbox_inches="tight")
plt.show()
