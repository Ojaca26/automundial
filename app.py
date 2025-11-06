# app.py

import streamlit as st
import pandas as pd
import re
import io
from typing import Optional
from sqlalchemy import text
from typing import Optional

# LangChain + Gemini
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain # Usaremos esta
import numpy as np # Asegúrate de que este import esté

# Micrófono en vivo (frontend) + fallback SR
from streamlit_mic_recorder import speech_to_text, mic_recorder
import speech_recognition as sr

# Agente de Correo
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import json

# ============================================
# 0) Configuración de la Página y Título
# ============================================
st.set_page_config(page_title="IANA para AutoLLantas", page_icon="logo_autollantas.png", layout="wide")

col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo_autollantas.png", width=120)
with col2:
    st.title("IANA: Tu Asistente IA para Análisis de Datos")
    st.markdown("Soy la red de agentes IA de **AutoLLantas**. Hazme una pregunta sobre los datos de tu negocio.")

# ============================================
# 1) Conexión a la Base de Datos y LLMs
# ============================================
@st.cache_resource
def get_database_connection():
    """
    Esta función se conecta a la base de datos usando los secretos de Streamlit.
    """
    with st.spinner("🛰️ Conectando a la base de datos..."):
        try:
            creds = st.secrets["db_credentials"]
            
            # 1. URI LIMPIA: Quitamos el parámetro ?ssl_mode de aquí.
            uri = f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}"
            
            # 2. ENGINE_ARGS: Pasamos la configuración de SSL aquí.
            engine_args = {
                "connect_args": {"ssl_disabled": True},
                # --- ⬇️ INICIO DE LA CORRECCIÓN ⬇️ ---
                # "pool_recycle" le dice al motor que descarte conexiones
                # más antiguas que X segundos (1800s = 30 minutos).
                # Esto previene el error "MySQL server has gone away".
                "pool_recycle": 1800 
                # --- ⬆️ FIN DE LA CORRECCIÓN ⬆️ ---
            }

            db = SQLDatabase.from_uri(uri, include_tables=["autollantas"], engine_args=engine_args)
            
            # Prueba la conexión
            db.run("SELECT 1")

            st.success("✅ Conexión a la base de datos establecida.")
            return db
        except Exception as e:
            st.error(f"Error al conectar a la base de datos: {e}")
            return None

@st.cache_resource
def get_llms():
    with st.spinner("🤝 Inicializando la red de agentes IANA..."):
        try:
            api_key = st.secrets["openai_api_key"]
            model_name = "gpt-4o"
            llm_sql = ChatOpenAI(model=model_name, temperature=0.1, openai_api_key=api_key)
            llm_analista = ChatOpenAI(model=model_name, temperature=0.1, openai_api_key=api_key)
            llm_orq = ChatOpenAI(model=model_name, temperature=0.0, openai_api_key=api_key)
            llm_validador = ChatOpenAI(model=model_name, temperature=0.0, openai_api_key=api_key)

            #api_key = st.secrets["google_api_key"]
            #common_config = dict(temperature=0.1, google_api_key=api_key)
            #llm_sql = ChatGoogleGenerativeAI(model="gemini-1.5-pro", **common_config)
            #llm_analista = ChatGoogleGenerativeAI(model="gemini-1.5-pro", **common_config)
            #llm_orq = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            #llm_validador = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            
            st.success("✅ Agentes de IANA listos.")
            return llm_sql, llm_analista, llm_orq, llm_validador
        except Exception as e:
            st.error(f"Error al inicializar los LLMs. Revisa tu API key. Detalle: {e}")
            return None, None, None, None

db = get_database_connection()
llm_sql, llm_analista, llm_orq, llm_validador = get_llms()


@st.cache_resource
def get_sql_agent(_llm, _db):
    if not _llm or not _db: return None
    with st.spinner("🛠️ Configurando agente SQL de IANA..."):
        toolkit = SQLDatabaseToolkit(db=_db, llm=_llm)
        
        # --- ⬇️ INICIO DE LA CORRECCIÓN ⬇️ ---
        agent = create_sql_agent(
            llm=_llm, 
            toolkit=toolkit, 
            verbose=False, 
            top_k=1000,
            # Esto le dice al agente que reintente si falla
            # al "parsear" la respuesta del LLM.
            handle_parsing_errors=True 
        )
        # --- ⬆️ FIN DE LA CORRECCIÓN ⬆️ ---
        
        st.success("✅ Agente SQL configurado.")
        return agent


agente_sql = get_sql_agent(llm_sql, db)

# ============================================
# 1.b) Reconocedor de Voz (fallback local)
# ============================================

@st.cache_resource
def get_recognizer():
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True
    return r

def transcribir_audio_bytes(data_bytes: bytes, language: str) -> Optional[str]:
    try:
        r = get_recognizer()
        with sr.AudioFile(io.BytesIO(data_bytes)) as source:
            audio = r.record(source)
        texto = r.recognize_google(audio, language=language)
        return texto.strip() if texto else None
    except Exception:
        return None

# ============================================
# 2) Agente de Correo (Lógica Mejorada)
# ============================================

def extraer_detalles_correo(pregunta_usuario: str) -> dict:
    st.info("🧠 El agente de correo está interpretando tu solicitud...")
    
    # Cargar la "agenda de contactos" desde los secretos
    contactos = dict(st.secrets.get("named_recipients", {}))
    default_recipient_name = st.secrets.get("email_credentials", {}).get("default_recipient", "")
    
    prompt = f"""
    Tu tarea es analizar la pregunta de un usuario y extraer los detalles para enviar un correo. Tu output DEBE SER un JSON válido.

    Agenda de Contactos Disponibles: {', '.join(contactos.keys())}

    Pregunta del usuario: "{pregunta_usuario}"

    Instrucciones para extraer:
    1.  `recipient_name`: Busca un nombre de la "Agenda de Contactos" en la pregunta. Si encuentras un nombre (ej: "Oscar"), pon ese nombre aquí. Si encuentras una dirección de correo explícita (ej: "test@test.com"), pon la dirección completa aquí. Si no encuentras ni nombre ni correo, usa "default".
    2.  `subject`: Crea un asunto corto y descriptivo basado en la pregunta.
    3.  `body`: Crea un cuerpo de texto breve y profesional para el correo.

    Ejemplo:
    Pregunta: "envía el reporte a Oscar por favor"
    JSON Output:
    {{
        "recipient_name": "Oscar",
        "subject": "Reporte de Datos Solicitado",
        "body": "Hola, como solicitaste, aquí tienes el reporte con los datos."
    }}
    
    JSON Output para la pregunta actual:
    """
    
    try:
        response = llm_analista.invoke(prompt).content
        json_response = response.strip().replace("```json", "").replace("```", "").strip()
        details = json.loads(json_response)
        
        recipient_identifier = details.get("recipient_name", "default")
        
        # Resolver el identificador a un correo real
        if "@" in recipient_identifier:
            final_recipient = recipient_identifier  # Ya es un correo
        elif recipient_identifier in contactos:
            final_recipient = contactos[recipient_identifier] # Buscar en la agenda
        else:
            final_recipient = default_recipient_name # Usar el por defecto

        return {
            "recipient": final_recipient,
            "subject": details.get("subject", "Reporte de Datos - IANA"),
            "body": details.get("body", "Adjunto encontrarás los datos solicitados.")
        }
    except Exception as e:
        st.warning(f"No pude interpretar los detalles del correo (error: {e}), usaré los valores por defecto.")
        return {
            "recipient": default_recipient_name,
            "subject": "Reporte de Datos - IANA",
            "body": "Adjunto encontrarás los datos solicitados."
        }


def enviar_correo_agente(recipient: str, subject: str, body: str, df: Optional[pd.DataFrame] = None):
    with st.spinner(f"📧 Enviando correo a {recipient}..."):
        try:
            creds = st.secrets["email_credentials"]
            sender_email = creds["sender_email"]
            sender_password = creds["sender_password"]
            
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            if df is not None and not df.empty:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                attachment = MIMEApplication(csv_buffer.getvalue(), _subtype='csv')
                attachment.add_header('Content-Disposition', 'attachment', filename="datos_iana.csv")
                msg.attach(attachment)
            
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            st.success(f"✅ Correo enviado exitosamente a {recipient}!")
            return {"texto": f"¡Listo! El correo fue enviado a {recipient}."}
            
        except Exception as e:
            st.error(f"❌ No se pudo enviar el correo. Error: {e}")
            return {"tipo": "error", "texto": f"Lo siento, no pude enviar el correo. Detalle del error: {e}"}

# ============================================
# 3) Funciones Auxiliares y Agentes (SIN CAMBIOS)
# ============================================
# (Todas las funciones desde _coerce_numeric_series hasta responder_conversacion se mantienen igual)
def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.replace(r'[\u00A0\s]', '', regex=True).str.replace(',', '', regex=False).str.replace('$', '', regex=False).str.replace('%', '', regex=False)
    try: return pd.to_numeric(s2)
    except Exception: return s
def get_history_text(chat_history: list, n_turns=3) -> str:
    if not chat_history or len(chat_history) <= 1: return ""
    history_text = []
    relevant_history = chat_history[-(n_turns * 2 + 1) : -1]
    for msg in relevant_history:
        content = msg.get("content", {}); text_content = ""
        if isinstance(content, dict): text_content = content.get("texto", "")
        elif isinstance(content, str): text_content = content
        if text_content:
            role = "Usuario" if msg["role"] == "user" else "IANA"
            history_text.append(f"{role}: {text_content}")
    if not history_text: return ""
    return "\n--- Contexto de Conversación Anterior ---\n" + "\n".join(history_text) + "\n--- Fin del Contexto ---\n"

def get_last_sql_from_history(chat_history: list) -> Optional[str]:
    """Extrae la última consulta SQL exitosa del historial."""
    st.info("Buscando la última consulta SQL en el historial...")
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            content = msg.get("content", {})
            if isinstance(content, dict) and content.get("sql"):
                st.success("¡Consulta anterior encontrada!")
                return content["sql"]
    st.warning("No se encontró una consulta SQL previa en el historial.")
    return None
    
def markdown_table_to_df(texto: str) -> pd.DataFrame:
    lineas = [l.rstrip() for l in texto.splitlines() if l.strip().startswith('|')]
    if not lineas: return pd.DataFrame()
    lineas = [l for l in lineas if not re.match(r'^\|\s*-{2,}', l)]
    filas = [[c.strip() for c in l.strip('|').split('|')] for l in lineas]
    if len(filas) < 2: return pd.DataFrame()
    header, data = filas[0], filas[1:]
    max_cols = len(header); data = [r + ['']*(max_cols - len(r)) if len(r) < max_cols else r[:max_cols] for r in data]
    df = pd.DataFrame(data, columns=header)
    for c in df.columns: df[c] = _coerce_numeric_series(df[c])
    return df
def _df_preview(df: pd.DataFrame, n: int = 5) -> str:
    if df is None or df.empty: return ""
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)
def interpretar_resultado_sql(res: dict) -> dict:
    df = res.get("df")
    if df is not None and not df.empty and res.get("texto") is None:
        if df.shape == (1, 1):
            valor = df.iloc[0, 0]; nombre_columna = df.columns[0]
            res["texto"] = f"La respuesta para '{nombre_columna}' es: **{valor}**"
            st.info("💡 Resultado numérico interpretado para una respuesta directa.")
    return res
    
def _asegurar_select_only(sql: str) -> str:
    sql_clean = sql.strip().rstrip(';')
    if not re.match(r'(?is)^\s*select\b', sql_clean): raise ValueError("Solo se permite ejecutar consultas SELECT.")
    sql_clean = re.sub(r'(?is)\blimit\s+\d+\s*$', '', sql_clean).strip()
    return sql_clean

def style_dataframe(df: pd.DataFrame):
    """Aplica el formato de miles y resalta la fila Total a un DataFrame."""
    
    # Detecta las columnas numéricas que no son ni año ni mes
    value_cols = [
        c for c in df.select_dtypes("number").columns
        if not re.search(r"(?i)\b(mes|año|dia|fecha)\b", c)
    ]
    
    # Define el diccionario de formato genérico de miles
    format_dict = {col: "{:,.0f}" for col in value_cols}

    # Define la función para resaltar la fila "Total"
    def highlight_total(row):
        # Revisa si la primera celda de la fila es "Total"
        if str(row.iloc[0]).lower() == "total":
            return ["font-weight: bold; background-color: #f8f9fa; border-top: 2px solid #999;"] * len(row)
        else:
            return [""] * len(row)

    # Aplica el resaltado y el formato
    try:
        return df.style.apply(highlight_total, axis=1).format(format_dict)
    except Exception:
        # Si algo falla, devuelve el df sin formato para evitar un crash
        return df

def limpiar_sql(sql_texto: str) -> str:
    """
    Limpia texto generado por LLM para dejar solo la consulta SQL válida.
    - Elimina prefijos como 'sql', 'sql:', 'SQL\n'
    - Elimina etiquetas ```sql``` o ``` ```
    - Recorta espacios y saltos de línea.
    """
    if not sql_texto:
        return ""

    # Elimina etiquetas markdown primero
    limpio = re.sub(r'```sql|```', '', sql_texto, flags=re.I)

    # Elimina cualquier prefijo 'sql' seguido de espacio, ':' o salto de línea
    limpio = re.sub(r'(?im)^\s*sql[\s:]+', '', limpio)

    # Busca el primer SELECT si todavía hay texto explicativo
    m = re.search(r'(?is)(select\b.+)$', limpio)
    if m:
        limpio = m.group(1)

    # Limpieza final
    return limpio.strip().rstrip(';')

# ESTA FUNCIÓN TAMBIÉN ES NECESARIA (la usa ejecutar_sql_real)
def _asegurar_select_only(sql: str) -> str:
    sql_clean = sql.strip().rstrip(';')
    if not re.match(r'(?is)^\s*select\b', sql_clean): raise ValueError("Solo se permite ejecutar consultas SELECT.")
    sql_clean = re.sub(r'(?is)\blimit\s+\d+\s*$', '', sql_clean).strip()
    return sql_clean



def ejecutar_sql_real(pregunta_usuario: str, hist_text: str, last_sql: Optional[str] = None):
    st.info("🤖 El agente de datos está traduciendo tu pregunta a SQL...")

    # --- Obtener Esquema ---
    try:
        # Obtener la info solo de la tabla 'autollantas'
        schema_info = db.get_table_info(table_names=["autollantas"])
    except Exception as e:
        st.error(f"Error crítico: No se pudo obtener el esquema de la tabla 'autollantas'. {e}")
        schema_info = "Error al obtener esquema. Asume columnas estándar."

    
    # --- ⬇️ INICIO DE LA MODIFICACIÓN DEL PROMPT (PASO 4) ⬇️ ---
    
    # Crear un bloque de "consulta anterior" solo si existe
    last_sql_context = ""
    if last_sql:
        last_sql_context = f"""
    --- CONSULTA ANTERIOR (Contexto) ---
    La última consulta SQL que ejecutaste fue:
    ```sql
    {last_sql}
    ```
    --- FIN CONSULTA ANTERIOR ---
    """

    # --- Crear Prompt (Asegúrate que el prompt esté completo aquí) ---
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL limpia (SOLO SELECT) sobre la tabla `autollantas` para responder la pregunta del usuario, basándote ESTRICTAMENTE en el siguiente esquema de tabla.

    --- ESQUEMA DE LA TABLA 'autollantas' ---
    {schema_info}
    --- FIN DEL ESQUEMA ---

    {last_sql_context}

    --- REGLAS DE MODIFICACIÓN (¡MUY IMPORTANTE!) ---
    1. Si la "Pregunta del usuario" parece ser una continuación o modificación de la "CONSULTA ANTERIOR" (ej: "agregale el mes", "ahora por cliente", "filtra solo Goodyear"), DEBES modificar esa consulta anterior.
    2. EJEMPLO: Si la consulta anterior fue `SELECT YEAR(_Fecha) as Año, SUM(Ventas_Reales) FROM autollantas WHERE YEAR(_Fecha) = 2025 GROUP BY YEAR(_Fecha)` y la pregunta nueva es "agregale el mes", la NUEVA consulta debe ser `SELECT YEAR(_Fecha) as Año, MONTH(_Fecha) as Mes, SUM(Ventas_Reales) FROM autollantas WHERE YEAR(_Fecha) = 2025 GROUP BY YEAR(_Fecha), MONTH(_Fecha)`.
    3. Si la pregunta es nueva (ej: "¿cuál es el costo total?"), IGNORA la consulta anterior y crea una nueva desde cero.

    --- REGLAS DE NEGOCIO Y FORMATO ---
    (Tus reglas existentes están intactas)
    <<< MANEJO DE "TOP N" >>>
    1. Si pide "top 5", "los 10 mejores", SIN agrupar -> Usa `ORDER BY ... DESC LIMIT N`.
    2. Si pide "top 5 de CADA mes", "por CADA línea" -> Usa `ROW_NUMBER()` con CTE.

     <<< CARTERA Y RECAUDO >>>
    1. Si el usuario pide algún tema de cartera como “cartera corriente” o “cartera vencida” o "cartera total", reponse con palabras profesionales, que no existe cartera en base de datos.

    <<< VALORES MONETARIOS >>>
    1. "margen"/"margen bruto": Usa `Porcentaje_Margen_Bruto` (relativo). Si pide "margen en pesos"/"absoluto" -> Usa `Margen_Bruto`.
    2. ❗ NUNCA USES AVG(Porcentaje_Margen_Bruto). Calcula dinámicamente: `(1 - SUM(Costo_Reales) / SUM(Ventas_Reales)) * 100` o `(SUM(Ventas_Reales - Costo_Reales) / SUM(Ventas_Reales)) * 100`.
    3. "% margen"/"margen porcentual": Usa el cálculo dinámico anterior o `Porcentaje_Margen_Bruto` si aplica directamente.
    4. "unidades vendidas": Usa `Unidades_Vendidas`.
    5. "precio promedio": Usa `Precio_Promedio`.
    6. "ventas"/"ventas totales": Usa `Ventas_Reales`.
    7. "costos": Usa `Costo_Reales`.

    <<< FILTRAR POR FECHA >>>
    # --- ⬇️ INICIO DE LA CORRECCIÓN ⬇️ ---
    1. Usa la columna `_Fecha`.
    2. Si pide año (ej: "2025") -> Añade `WHERE YEAR(_Fecha) = [año]`.
    # --- ⬆️ FIN DE LA CORRECCIÓN ⬆️ ---

    <<< BÚSQUEDA DE PRODUCTOS/CLIENTES/MARCAS >>>
    1. "artículo"/"producto": Usa `WHERE LOWER(Nombre_Articulo) LIKE '%palabra%'`.
    2. "cliente": Usa `WHERE LOWER(Nombre_Cliente) LIKE '%palabra%'`.
    3. "línea"/"marca" (Goodyear, etc.): Usa `WHERE LOWER(Nombre_Linea) LIKE '%marca%'`.

    --- CONTEXTO Y PREGUNTA ---
    {hist_text}
    Pregunta del usuario: "{pregunta_usuario}"

    --- SALIDA ---
    Devuelve SOLO la consulta SQL (sin explicaciones, sin markdown ```sql```).
    """ # Fin del prompt
    
    # --- ⬆️ FIN DE LA MODIFICACIÓN DEL PROMPT (PASO 4) ⬆️ ---

    try:
        # --- Usar SQLDatabaseChain para GENERAR SQL ---
        # NOTA: Con estas versiones antiguas, es posible que 'return_sql' no exista,
        # pero 'SQLDatabaseChain' por defecto ponía el SQL en 'intermediate_steps'.
        # El método .invoke() es de versiones más nuevas, usaremos .run()
        
        # Con la versión 0.2.1, es mejor usar la llamada al LLM directa que funcionó en Ventus.
        # Vamos a replicar el método de VENTUS (LLM.invoke) que es más seguro.
        
        # ⬇️ --- REPLICANDO EL MÉTODO DE VENTUS --- ⬇️
        sql_query_bruta = llm_sql.invoke(prompt_con_instrucciones).content
        # ⬆️ --- FIN DEL MÉTODO DE VENTUS --- ⬆️

        if not sql_query_bruta:
            st.error("La cadena SQL no devolvió una consulta SQL válida.")
            return {"sql": None, "df": None, "error": "No se generó SQL."}

        st.text_area("🧩 SQL generado por el modelo:", sql_query_bruta, height=100)

        # --- Limpieza del SQL ---
        sql_query_limpia = limpiar_sql(sql_query_bruta) # Usa tu función limpiar_sql

        if not sql_query_limpia.lower().startswith("select"):
            m = re.search(r'(?is)(select\b.+)$', sql_query_limpia)
            if m:
                sql_query_limpia = m.group(1).strip()

        sql_query_limpia = _asegurar_select_only(sql_query_limpia)

        if not sql_query_limpia:
            st.error("El SQL generado quedó vacío después de la limpieza.")
            return {"sql": None, "df": None, "error": "SQL vacío tras limpieza."}


        st.code(sql_query_limpia, language="sql")

        # --- Ejecución del SQL ---
        with st.spinner("⏳ Ejecutando consulta..."):
            with db._engine.connect() as conn:
                df = pd.read_sql(text(sql_query_limpia), conn)

        st.success(f"✅ ¡Consulta ejecutada! Filas: {len(df)}")

        # --- Post-procesamiento (Completo y Corregido) ---
        value_cols = [] # Definir fuera del try para tenerla disponible
        try:
            if not df.empty:
                # Añadir columna Año si es relevante
                year_match = re.search(r"YEAR\([^)]*\)\s*=\s*(\d{4})", sql_query_limpia)
                year_value = year_match.group(1) if year_match else None
                if year_value and "Año" not in df.columns:
                    df.insert(0, "Año", year_value)

                # Identificar columnas de valor numérico (excluyendo fecha/tiempo)
                value_cols = [
                    c for c in df.select_dtypes("number").columns
                    if not re.search(r"(?i)\b(mes|año|dia|fecha|id|codigo)\b", c)
                ]

                # Añadir fila de Total (si hay más de una fila y columnas de valor)
                if value_cols and len(df) > 1:
                    total_row = {}
                    for col in df.columns:
                        if col in value_cols:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                total_row[col] = df[col].sum()
                            else:
                                total_row[col] = np.nan
                        elif pd.api.types.is_numeric_dtype(df[col]):
                            total_row[col] = np.nan
                        else:
                            total_row[col] = ""

                    first_col_name = df.columns[0]
                    total_row[first_col_name] = "Total"
                    
                    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

                
                # --- ⬇️ INICIO DE LA SOLUCIÓN (LÍMITE DE CELDAS) ⬇️ ---
                
                # Límite de celdas (el default de Streamlit es 262144)
                MAX_CELLS_FOR_STYLING = 262144 

                # df.size es el total de celdas (filas * columnas)
                if df.size > MAX_CELLS_FOR_STYLING:
                    st.warning(f"⚠️ La tabla resultante es muy grande ({df.size:,.0f} celdas) para aplicar formatos. Se mostrarán los datos en modo simple.")
                    # Devolvemos solo el df crudo, SIN la clave "styled"
                    # La función 'procesar_pregunta' usará este 'df' crudo
                    return {"sql": sql_query_limpia, "df": df}
                
                st.info("🎨 Aplicando formato a la tabla...")
                # --- ⬆️ FIN DE LA SOLUCIÓN (LÍMITE DE CELDAS) ⬆️ ---


                # --- Aplicar Estilos ---
                def highlight_total(row):
                    if isinstance(row.iloc[0], str) and row.iloc[0].lower() == "total":
                        return ["font-weight: bold; background-color: #f8f9fa; border-top: 2px solid #999;"] * len(row)
                    else:
                        return [""] * len(row)

                styled_df = df.style.apply(highlight_total, axis=1)


                # --- ⬇️ INICIO DE FORMATO SEGURO (Previene crash con 'Total') ⬇️ ---
        
                # 1. Define un formateador "seguro" que ignora strings (como "Total")
                def safe_num_format(val, fmt_str):
                    if pd.api.types.is_number(val):
                        try:
                            return fmt_str.format(val)
                        except (ValueError, TypeError):
                            return val # Fallback si el formato falla
                    return val # Retorna valores no numéricos (como "Total") tal cual

                # 2. Crea un mapa de las "cadenas de formato" que queremos
                format_map_strings = {}
                
                # Formato para columnas de valor (ej. Ventas_Reales)
                for col in value_cols:
                    format_map_strings[col] = "{:,.0f}"

                # Formato para columnas específicas (ej. Año, Mes)
                if "Mes" in df.columns:
                    format_map_strings["Mes"] = "{:.0f}"
                if "Año" in df.columns:
                    # Esta columna puede tener "Total" asi que el safe_format es vital
                    format_map_strings["Año"] = "{:.0f}" 
                
                # Formato para porcentajes (basado en el prompt de 'autollantas')
                percent_cols = [col for col in df.columns if "porcentaje" in col.lower() or "%" in col.lower() or col == "Porcentaje_Margen_Bruto"]
                for col in percent_cols:
                    format_map_strings[col] = "{:,.2f}%"

                # 3. Crea el diccionario final de formateadores "callable" (funciones)
                final_formatter = {}
                for col, fmt_str in format_map_strings.items():
                    # Creamos una función lambda "segura" para cada columna
                    final_formatter[col] = lambda val, f=fmt_str: safe_num_format(val, f)

                # 4. Aplica el diccionario de formateadores seguros
                styled_df = styled_df.format(final_formatter, na_rep="")
                
                # --- ⬆️ FIN DE FORMATO SEGURO ⬆️ ---

                return {"sql": sql_query_limpia, "df": df, "styled": styled_df}

        except Exception as e:
            st.warning(f"⚠️ No se pudo aplicar formato ni totales: {e}")
            return {"sql": sql_query_limpia, "df": df}

    # --- Manejo de error general ---
    except Exception as e:
        st.warning(f"❌ Error en la consulta directa. Intentando método alternativo... Detalle: {e}")
        return {"sql": None, "df": None, "error": str(e)}


def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str, hist_text: str):
    st.info("🤔 Activando el agente SQL experto como plan B (con instrucciones mejoradas)...")

    # --- ⬇️ INICIO DE LA CORRECCIÓN ⬇️ ---
    # El 'agente_sql' es un AgentExecutor, funciona mejor con una
    # pregunta simple, no con un prompt de reglas complejo.
    
    # Eliminamos el 'prompt_con_instrucciones' gigante que estaba aquí.
    
    simple_prompt = f"""
    {hist_text}
    
    Usando las herramientas de base de datos disponibles, responde la siguiente pregunta del usuario.
    
    Pregunta: "{pregunta_usuario}"
    """
    # --- ⬆️ FIN DE LA CORRECCIÓN ⬆️ ---

    try:
        with st.spinner("💬 Pidiendo al agente SQL experto que responda..."):
            
            # --- ⬇️ INICIO DE LA CORRECCIÓN ⬇️ ---
            # Usamos el nuevo prompt simple
            res = agente_sql.invoke(simple_prompt)
            # --- ⬆️ FIN DE LA CORRECCIÓN ⬆️ ---
            
            texto_salida = res["output"] if isinstance(res, dict) and "output" in res else str(res)
        
        st.info("📝 Intentando convertir la respuesta en una tabla de datos...");
        df_md = markdown_table_to_df(texto_salida)
        
        if df_md.empty: 
            st.warning("La conversión de Markdown a tabla no produjo filas. Se mostrará la salida cruda.")
            return {"texto": texto_salida, "df": df_md}

        # Generamos un texto introductorio para que no se vea vacío
        resumen_texto = f"Entendido. Aquí tienes la respuesta a tu pregunta sobre '{pregunta_usuario}' generada por el agente experto:"
        return {"texto": resumen_texto, "df": df_md}
        
    except Exception as e:
        st.error(f"❌ El agente SQL experto también encontró un problema: {e}")
        return {"texto": f"[SQL_ERROR] {e}", "df": pd.DataFrame()}

        
def analizar_con_datos(pregunta_usuario: str, hist_text: str, df: pd.DataFrame | None, feedback: str = None):
    st.info("\n🧠 El analista experto está examinando los datos...")
    correccion_prompt = ""
    if feedback:
        st.warning(f"⚠️ Reintentando con feedback: {feedback}")
        correccion_prompt = (f'INSTRUCCIÓN DE CORRECCIÓN: Tu respuesta anterior fue incorrecta. Feedback: "{feedback}". Genera una NUEVA respuesta corrigiendo este error.')
    preview = _df_preview(df, 500) or "(sin datos en vista previa; verifica la consulta)"
    prompt_analisis = f"""{correccion_prompt}\nEres IANA, un analista de datos senior EXTREMADAMENTE PRECISO y riguroso.\n---\n<<< REGLAS CRÍTICAS DE PRECISIÓN >>>\n1. **NO ALUCINAR**: NUNCA inventes números, totales, porcentajes o nombres de productos/categorías que no estén EXPRESAMENTE en la tabla de 'Datos'.\n2. **DATOS INCOMPLETOS**: Reporta los vacíos (p.ej., "sin datos para Marzo") sin inventar valores.\n3. **VERIFICAR CÁLCULOS**: Antes de escribir un número, revisa el cálculo (sumas/conteos/promedios) con los datos.\n4. **CITAR DATOS**: Basa CADA afirmación que hagas en los datos visibles en la tabla.\n---\nPregunta Original: {pregunta_usuario}\n{hist_text}\nDatos para tu análisis (usa SÓLO estos):\n{preview}\n---\nFORMATO OBLIGATORIO:\n📌 Análisis Ejecutivo de datos:\n1. Calcular totales y porcentajes clave.\n2. Detectar concentración.\n3. Identificar patrones temporales.\n4. Analizar dispersión.\nEntregar el resultado en 3 bloques:\n📌 Resumen Ejecutivo: hallazgos principales con números.\n🔍 Números de referencia: totales, promedios, ratios.\n⚠ Importante: Sé muy breve, directo y diciente."""
    with st.spinner("💡 Generando análisis avanzado..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis
    
def responder_conversacion(pregunta_usuario: str, hist_text: str):
    st.info("💬 Activando modo de conversación...")
    prompt_personalidad = f"""Tu nombre es IANA, una IA amable de autollantas. Ayuda a analizar datos.\nSi el usuario hace un comentario casual, responde amablemente de forma natural, muy humana y redirígelo a tus capacidades.\n{hist_text}\nPregunta: "{pregunta_usuario}" """
    respuesta = llm_analista.invoke(prompt_personalidad).content
    return {"texto": respuesta, "df": None, "analisis": None}

def generar_resumen_tabla(pregunta_usuario: str, res: dict) -> dict:
    st.info("✍️ Generando un resumen introductorio para la tabla...")
    df = res.get("df")
    if df is None or df.empty:
        return res

    # --- INICIO DE LA MODIFICACIÓN DEL PROMPT ---
    prompt = f"""
    Actúa como IANA, un analista de datos amable y servicial.
    Tu tarea es escribir una breve y conversacional introducción para la tabla de datos que estás a punto de mostrar.
    Basa tu respuesta en la pregunta del usuario para que se sienta como una continuación natural de la conversación.
    Si la respuesta no le gustó al USUARIO, disculpate es posible que le entendiste mal.
    
    IMPORTANTE: Varía tus respuestas. No uses siempre la misma frase "Por supuesto". Suena natural y humana.

    Pregunta del usuario: "{pregunta_usuario}"
    
    ---
    Aquí tienes varios ejemplos de cómo responder:

    Ejemplo 1:
    Pregunta: "cuáles son los proveedores"
    Respuesta: "¡Listo! Aquí tienes la lista de proveedores que encontré:"

    Ejemplo 2:
    Pregunta: "y sus ventas?"
    Respuesta: "He consultado las cifras de ventas. Te las muestro en la siguiente tabla:"

    Ejemplo 3:
    Pregunta: "y en q % esta su consumo?"
    Respuesta: "Perfecto, aquí está el desglose de los porcentajes de consumo que pediste:"

    Ejemplo 4:
    Pregunta: "dame el total por mes"
    Respuesta: "Claro que sí. He preparado la tabla con los totales por mes:"
    ---

    Ahora, genera la introducción para la pregunta del usuario actual:
    """
    # --- FIN DE LA MODIFICACIÓN DEL PROMPT ---
    try:
        introduccion = llm_analista.invoke(prompt).content
        res["texto"] = introduccion
    except Exception as e:
        st.warning(f"No se pudo generar el resumen introductorio. Error: {e}")
        res["texto"] = "Aquí están los datos que solicitaste:"
    return res

# ============================================
# 4) Orquestador y Validación
# ============================================
def validar_y_corregir_respuesta_analista(pregunta_usuario: str, res_analisis: dict, hist_text: str) -> dict:
    MAX_INTENTOS = 2
    for intento in range(MAX_INTENTOS):
        st.info(f"🕵️‍♀️ Supervisor de Calidad: Verificando análisis (Intento {intento + 1})..."); contenido_respuesta = res_analisis.get("analisis", "") or ""
        if not contenido_respuesta.strip(): return {"tipo": "error", "texto": "El análisis generado estaba vacío."}
        df_preview = _df_preview(res_analisis.get("df"), 500) or "(sin vista previa de datos)"
        prompt_validacion = f"""Eres un supervisor de calidad estricto. Valida si el 'Análisis' se basa ESTRICTAMENTE en los 'Datos de Soporte'.\nFORMATO:\n- Si está 100% basado en los datos: APROBADO\n- Si alucina/inventa/no es relevante: RECHAZADO: [razón corta y accionable]\n---\nPregunta: "{pregunta_usuario}"\nDatos de Soporte:\n{df_preview}\n---\nAnálisis a evaluar:\n\"\"\"{contenido_respuesta}\"\"\"\n---\nEvaluación:"""
        try:
            resultado = llm_validador.invoke(prompt_validacion).content.strip(); up = resultado.upper()
            if up.startswith("APROBADO"):
                st.success("✅ Análisis aprobado por el Supervisor."); return res_analisis
            elif up.startswith("RECHAZADO"):
                feedback_previo = resultado.split(":", 1)[1].strip() if ":" in resultado else "Razón no especificada."
                st.warning(f"❌ Análisis rechazado. Feedback: {feedback_previo}")
                if intento < MAX_INTENTOS - 1:
                    st.info("🔄 Regenerando análisis con feedback...")
                    res_analisis["analisis"] = analizar_con_datos(pregunta_usuario, hist_text, res_analisis.get("df"), feedback=feedback_previo)
                else: return {"tipo": "error", "texto": "El análisis no fue satisfactorio incluso después de una corrección."}
            else: return {"tipo": "error", "texto": f"Respuesta ambigua del validador: {resultado}"}
        except Exception as e: return {"tipo": "error", "texto": f"Excepción durante la validación: {e}"}
    return {"tipo": "error", "texto": "Se alcanzó el límite de intentos de validación."}


def clasificar_intencion(pregunta: str) -> str:
    # <<< MODIFICADO para mejorar la detección de consultas de datos >>>
    prompt_orq = f"""
Clasifica la intención del usuario en UNA SOLA PALABRA. Presta especial atención a los verbos de acción y palabras clave.

1. `analista`: Si la pregunta pide explícitamente una interpretación, resumen, comparación o explicación.
   PALABRAS CLAVE PRIORITARIAS: analiza, compara, resume, explica, por qué, tendencia, insights, dame un análisis, haz un resumen, interpreta.
   Si una de estas palabras clave está presente, la intención SIEMPRE es `analista`.

2. `consulta`: Si la pregunta pide datos crudos (listas, conteos, totales, valores, métricas) o resultados numéricos directos, y NO contiene palabras clave de `analista`.
   Ejemplos: 'cuántos proveedores hay', 'lista todos los productos', 'muéstrame el total', 'ventas por mes', 'margen por cliente', 'costo total', 'precio promedio'.
   PALABRAS CLAVE ADICIONALES: venta, ventas, costo, costos, margen, precio, unidades, rubro, cliente, artículo, producto, línea, familia, total, facturado, utilidad.

3. `correo`: Si la pregunta pide explícitamente enviar un correo, email o reporte.
   PALABRAS CLAVE: envía, mandar, correo, email, reporte a, envíale a.

4. `conversacional`: Si es un saludo o una pregunta general no relacionada con datos.
   Ejemplos: 'hola', 'gracias', 'qué puedes hacer', 'cómo estás'.

Pregunta: "{pregunta}"
Clasificación:
"""

    try:
        opciones = {"consulta", "analista", "conversacional", "correo"}
        r = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")

        # 🔹 Refuerzo semántico (fallback inteligente)
        # Si contiene palabras típicas de consultas de datos, fuerza 'consulta'
        if any(pal in pregunta.lower() for pal in [
            "venta", "ventas", "margen", "costo", "costos", "precio", "unidades",
            "rubro", "cliente", "artículo", "producto", "línea", "familia", "total", "facturado"
        ]):
            return "consulta"

        # Si el modelo devuelve algo fuera de las opciones válidas → se asume consulta
        if r not in opciones:
            return "consulta"

        return r

    except Exception:
        return "consulta"

def obtener_datos_sql(pregunta_usuario: str, hist_text: str, last_sql: Optional[str] = None) -> dict:
    """
    Esta función SIEMPRE ejecuta una nueva consulta SQL.
    La lógica para re-usar tablas anteriores ahora vive en el orquestador.
    """
    
    # --- ⬇️ INICIO DE LA MODIFICACIÓN ⬇️ ---
    # 3. Pasa 'last_sql' a la función de ejecución real
    res_real = ejecutar_sql_real(pregunta_usuario, hist_text, last_sql)
    # --- ⬆️ FIN DE LA MODIFICACIÓN ⬆️ ---
    
    if res_real.get("df") is not None and not res_real["df"].empty:
        return res_real
        
    # Fallback al agente de lenguaje natural si la ejecución directa falla
    st.warning("La consulta directa falló, intentando con el agente de lenguaje natural...")
    return ejecutar_sql_en_lenguaje_natural(pregunta_usuario, hist_text)

def guardian_agent(pregunta_usuario: str, sql_propuesta: str | None = None) -> bool:
    """
    Revisa que la pregunta y la consulta SQL no representen un riesgo.
    Devuelve True si es segura, False si debe bloquearse.
    """
    st.info("🧩 Guardian Agent: verificando seguridad de la solicitud...")

    # --- Nivel 1: Palabras prohibidas SQL (Se mantiene igual) ---
    palabras_peligrosas = [
        "drop", "delete", "truncate", "update", "insert", "alter",
        "create", "replace", "grant", "revoke", "commit", "rollback",
        "information_schema", "mysql", "sys", "pg_", "dual"
    ]
    if sql_propuesta:
        sql_lower = sql_propuesta.lower()
        if any(p in sql_lower for p in palabras_peligrosas):
            st.error("🚫 Guardian Agent: consulta SQL insegura detectada (palabras prohibidas).")
            return False

    # --- Nivel 2: Análisis semántico de la pregunta ---
    # --- ⬇️ HEMOS MODIFICADO ESTE PROMPT ⬇️ ---
    prompt_guardian = f"""
Eres un agente de seguridad y cumplimiento. Tu tarea es revisar si la siguiente pregunta o consulta del usuario podría
implicar acceso a información sensible, manipulación de datos o riesgo de fuga de privacidad.

Pregunta del usuario: "{pregunta_usuario}"

Reglas:
1. Bloquea solo si pide datos personales (correos, teléfonos, NIT, direcciones, contraseñas, claves, API keys).
2. Bloquea si intenta modificar datos con verbos peligrosos (eliminar, borrar, actualizar, insertar, modificar, crear, drop, alter).
3. Bloquea si pide estructura interna del sistema o base de datos sensible.
4. PERMITE expresamente solicitudes de análisis financiero (márgenes, ventas, costos, totales, promedios, cantidades, precios).
5. **NUEVA REGLA:** PERMITE verbos analíticos comunes (ej. "agrega la columna mes", "agrupa por", "compara con", "incluye el cliente") ya que se refieren a la *presentación* de los datos (un SELECT), no a la *modificación* de la base de datos (un UPDATE/INSERT).

Responde solo con una palabra:
- "APROBADO" si es seguro.
- "BLOQUEADO" si no lo es.
    """
    # --- ⬆️ FIN DE LA MODIFICACIÓN ⬆️ ---
    
    try:
        decision = llm_validador.invoke(prompt_guardian).content.strip().upper()
        if "BLOQUEADO" in decision:
            st.error("🚫 Guardian Agent: solicitud bloqueada por seguridad.")
            return False
    except Exception as e:
        st.warning(f"Guardian Agent no pudo validar la solicitud ({e}), continuaré con precaución.")
        return True

    st.success("✅ Guardian Agent: solicitud aprobada.")
    return True

def orquestador(pregunta_usuario: str, chat_history: list):
    with st.expander("⚙️ Ver Proceso de IANA", expanded=False):
        hist_text = get_history_text(chat_history)
        
        # --- ⬇️ INICIO DE LA MODIFICACIÓN (Paso 2.1) ⬇️ ---
        # 1. Obtener el último SQL exitoso del historial
        last_sql = get_last_sql_from_history(chat_history)
        # --- ⬆️ FIN DE LA MODIFICACIÓN ⬆️ ---

        clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ ¡Intención detectada! Tarea: {clasificacion.upper()}.")

        if clasificacion == "conversacional":
            return responder_conversacion(pregunta_usuario, hist_text)
        
        if clasificacion == "correo":
            df_para_enviar = None
            for msg in reversed(st.session_state.get('messages', [])):
                if msg.get('role') == 'assistant':
                    content = msg.get('content', {}); df_prev = content.get('df')
                    if isinstance(df_prev, pd.DataFrame) and not df_prev.empty:
                        df_para_enviar = df_prev
                        st.info("📧 Datos de la tabla anterior encontrados para adjuntar al correo.")
                        break
            
            if df_para_enviar is None:
                st.warning("No encontré una tabla en la conversación reciente para enviar. El correo irá sin datos adjuntos.")

            detalles = extraer_detalles_correo(pregunta_usuario)
            return enviar_correo_agente(
                recipient=detalles["recipient"],
                subject=detalles["subject"],
                body=detalles["body"],
                df=df_para_enviar
            )
            
        if not guardian_agent(pregunta_usuario):
            return {"tipo": "error", "texto": "🚫 Solicitud bloqueada por el agente guardián por motivos de seguridad."}
        

        # --- ⬇️ INICIO DE LA MODIFICACIÓN (Paso 2.2) ⬇️ ---
        # Reemplazamos la simple llamada a 'obtener_datos_sql' por esta lógica de contexto

        res_datos = {}
        df_previo = None
        use_previous_df = False

        # 1. Buscar si ya existe una tabla en la conversación
        for msg in reversed(st.session_state.get('messages', [])):
            if msg.get('role') == 'assistant':
                content = msg.get('content', {}); df_prev = content.get('df')
                if isinstance(df_prev, pd.DataFrame) and not df_prev.empty:
                    df_previo = df_prev
                    break
        
        # 2. Decidir si usamos la tabla anterior
        if df_previo is not None:
            prompt_lower = pregunta_usuario.lower()
            prompt_clean = prompt_lower.strip().rstrip("?.!")
            
            # Keywords que SIEMPRE se refieren a la tabla anterior
            contextual_keywords = ["anterior", "esos datos", "esa tabla", "la tabla"]
            # Triggers de análisis simple
            simple_analysis_triggers = ["analiza", "analisis", "análisis", "haz un analisis", "dame un analisis"]

            if any(keyword in prompt_lower for keyword in contextual_keywords):
                use_previous_df = True
            elif clasificacion == "analista" and prompt_clean in simple_analysis_triggers:
                use_previous_df = True
        
        # 3. Obtener los datos basándose en la decisión
        if use_previous_df:
            st.info("💡 Usando la tabla anterior para la nueva solicitud.")
            # Pasamos tanto el 'df' previo como el 'last_sql'
            res_datos = {"df": df_previo, "sql": last_sql}
        else:
            # Esta es la llamada que teníamos antes, pero ahora le pasamos 'last_sql'
            res_datos = obtener_datos_sql(pregunta_usuario, hist_text, last_sql)
        
        # --- ⬆️ FIN DE LA MODIFICACIÓN ⬆️ ---


        if res_datos.get("df") is None or res_datos["df"].empty:
            return {"tipo": "error", "texto": "Lo siento, no pude obtener datos para tu pregunta. Intenta reformularla."}

        # if clasificacion == "consulta":
        #     st.success("✅ Consulta directa completada.")
        #     return interpretar_resultado_sql(res_datos)

        if clasificacion == "consulta":
            st.success("✅ Consulta directa completada.")
            # Primero, intentamos interpretar el resultado como siempre
            res_interpretado = interpretar_resultado_sql(res_datos)
    
            # Luego, si no se generó texto (porque es una tabla), creamos la introducción
            if res_interpretado.get("texto") is None and res_interpretado.get("df") is not None and not res_interpretado["df"].empty:
                res_interpretado = generar_resumen_tabla(pregunta_usuario, res_interpretado)
    
            return res_interpretado

        if clasificacion == "analista":
            st.info("🧠 Generando análisis inicial...")
            res_datos["analisis"] = analizar_con_datos(pregunta_usuario, hist_text, res_datos.get("df"))
            return validar_y_corregir_respuesta_analista(pregunta_usuario, res_datos, hist_text)

# ============================================
# 5) Interfaz: Micrófono en vivo + Chat
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": {"texto": "¡Hola! Soy IANA, tu asistente de IA de autollantas. ¿Qué te gustaría saber?"}}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message.get("content", {});
        if isinstance(content, dict):
            if content.get("texto"): st.markdown(content["texto"])
            
            if content.get("styled") is not None: st.dataframe(content["styled"])
            elif isinstance(content.get("df"), pd.DataFrame) and not content["df"].empty:
                styled_df = style_dataframe(content["df"])
                st.dataframe(styled_df)
            
            if content.get("analisis"): st.markdown(content["analisis"])
            
        elif isinstance(content, str): st.markdown(content)

st.markdown("### 🎤 Habla con IANA o escribe tu pregunta")
lang = st.secrets.get("stt_language", "es-CO")

# Unificar el procesamiento de la pregunta
def procesar_pregunta(prompt):
    if prompt:
        if not all([db, llm_sql, llm_analista, llm_orq, agente_sql, llm_validador]):
            st.error("La aplicación no está completamente inicializada. Revisa los errores de conexión o de API key.")
            return

        st.session_state.messages.append({"role": "user", "content": {"texto": prompt}})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            res = orquestador(prompt, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": res})

        if res and res.get("tipo") != "error":
            if res.get("texto"): st.markdown(res["texto"])
            
            # --- ⬇️ INICIO DE LA CORRECCIÓN ⬇️ ---
            # Revisa si existe la versión "styled" (con formato)
            if res.get("styled") is not None:
                st.dataframe(res["styled"])
            # Si no, muestra la versión "cruda" (df)
            elif isinstance(res.get("df"), pd.DataFrame) and not res["df"].empty:
                st.dataframe(res["df"])
            # --- ⬆️ FIN DE LA CORRECCIÓN ⬆️ ---
            
            if res.get("analisis"):
                st.markdown("---"); st.markdown("### 🧠 Análisis de IANA"); st.markdown(res["analisis"])
                st.toast("Análisis generado ✅", icon="✅")
        elif res:
            st.error(res.get("texto", "Ocurrió un error inesperado."))
            st.toast("Hubo un error ❌", icon="❌")

# Contenedor para los inputs
input_container = st.container()
with input_container:
    col1, col2 = st.columns([1, 4])
    with col1:
        voice_text = speech_to_text(language=lang, start_prompt="🎙️ Hablar", stop_prompt="🛑 Detener", use_container_width=True, just_once=True, key="stt")
    with col2:
        prompt_text = st.chat_input("... o escribe tu pregunta aquí")

# Determinar qué prompt usar
prompt_a_procesar = None
if voice_text:
    prompt_a_procesar = voice_text
elif prompt_text:
    prompt_a_procesar = prompt_text

# Procesar el prompt si existe
if prompt_a_procesar:
    procesar_pregunta(prompt_a_procesar)
    
