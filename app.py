# app.py

import streamlit as st
import pandas as pd
import re
import io
from typing import Optional
from sqlalchemy import text

# LangChain + Gemini
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain

#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_community.utilities import SQLDatabase
#from langchain.agents import create_sql_agent
#from langchain.agents.agent_toolkits import SQLDatabaseToolkit
#from langchain.chains import create_sql_query_chain

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
st.set_page_config(page_title="IANA para Automundial", page_icon="logo_automundial.png", layout="wide")

col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo_automundial.png", width=120)
with col2:
    st.title("IANA: Tu Asistente IA para Análisis de Datos")
    st.markdown("Soy la red de agentes IA de **Automundial**. Hazme una pregunta sobre los datos de tu negocio.")

# ============================================
# 1) Conexión a la Base de Datos y LLMs
# ============================================

@st.cache_resource
def get_database_connection():
    with st.spinner("🛰️ Conectando a la base de datos de automundial..."):
        try:
            creds = st.secrets["db_credentials"]
            uri = f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}"
            engine_args = {"pool_recycle": 3600, "pool_pre_ping": True}
            db = SQLDatabase.from_uri(uri, include_tables=["automundial"], engine_args=engine_args)
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
        agent = create_sql_agent(llm=_llm, toolkit=toolkit, verbose=False, top_k=1000)
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

def ejecutar_sql_real(pregunta_usuario: str, hist_text: str):
    st.info("🤖 El agente de datos está traduciendo tu pregunta a SQL...")
    
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL limpia (SOLO SELECT) sobre la tabla `automundial` para responder la pregunta del usuario.

    ---
    <<< NUEVA REGLA: PARA VALORES MONETARIOS >>>
     1. Cuando el usuario mencione “margen”, “margen bruto” o “ganancia bruta”, se debe consultar la información en la columna 'Porcentaje_Margen_Bruto', que representa el **margen relativo** (porcentaje de utilidad sobre ventas).  
       Si el usuario pide explícitamente “margen en pesos”, “margen monetario” o “margen absoluto”, entonces usa la columna 'Margen_Bruto', que representa el **margen absoluto** (valor monetario de la utilidad bruta).  
       Ejemplo:  
       - “Dame el margen bruto por mes” → usa `Porcentaje_Margen_Bruto`  
       - “Dame el margen absoluto en pesos” → usa `Margen_Bruto`
    2. Cuando el usuario mencione “porcentaje de margen”, “% margen”, “margen porcentual” o “margen en porcentaje”, se debe consultar la información en la columna 'Porcentaje_Margen_Bruto', que representa la proporción del margen bruto sobre las ventas reales.
    3. Cuando el usuario mencione “unidades vendidas”, “cantidad de productos vendidos” o “número de ventas”, se está refiriendo al campo 'Unidades_Vendidas'.
    4. Cuando el usuario pregunte por “precio promedio”, “valor medio de venta” o “promedio de precios”, se refiere al campo 'Precio_Promedio', que corresponde al promedio del valor unitario de las ventas.
    5. Cuando el usuario mencione “ventas reales”, “ventas totales” o “valor vendido”, se está refiriendo al campo 'Ventas_Reales', que representa el total monetario facturado o reconocido como ingreso real.
    6. Cuando el usuario mencione “costos reales”, “costos totales” o “valor del costo”, se refiere al campo Costo_Reales, que muestra el total de costos asociados a las ventas (sin incluir margen ni impuestos).
    7. Ejemplo: Si la pregunta es "¿cuál es el total facturado?", la consulta debería ser algo como `SELECT SUM(Ventas_Reales) FROM automundial;`. Aplica este patrón a otras métricas.
    ---
    <<< REGLA CRÍTICA PARA FILTRAR POR FECHA >>>
    1. Tu tabla tiene una columna de fecha llamada `Fecha`.
    2. Si el usuario especifica un año (ej: "del 2025", "en 2024"), SIEMPRE debes añadir una condición `WHERE YEAR(Fecha) = [año]` a la consulta.
    3. Ejemplo: "dame las ventas de 2025" -> DEBE INCLUIR `WHERE YEAR(Fecha) = 2025`.
    ---
    <<< REGLA DE ORO PARA BÚSQUEDA DE PRODUCTOS >>>
    1. Cuando el usuario mencione “artículo”, “producto”, “ítem”, “referencia”, “nombre del repuesto” o “nombre del material”, se está refiriendo al campo 'Nombre_Articulo', el cual contiene el nombre comercial o técnico de cada producto registrado en inventario o en las órdenes.
        Este campo puede incluir detalles como:
        - Medidas o especificaciones (ej. 195/60R16, 11R-22.5)
        - Marca o fabricante (ej. Yokohama, Firestone, Alliance)
        - Tipo o aplicación (ej. filtro de combustible, llanta, aire, repuesto)
    2. Si el usuario pregunta por un producto específico, usa `WHERE LOWER(Nombre_Articulo) LIKE '%palabra%'.
    3. Cuando el usuario mencione “cliente”, “empresa”, “razón social”, “comprador”, “contratante” o “nombre del cliente”, se está refiriendo al campo 'Nombre_Cliente', que representa la entidad (persona natural o jurídica) a la que se le vendió, facturó o prestó un servicio.
    4. Cuando el usuario mencione “línea”, “marca”, “familia de producto”, “referencia comercial” o “proveedor principal”, se está refiriendo al campo 'Nombre_Linea', el cual identifica la marca, línea o categoría principal a la que pertenece un artículo.
    ---
    {hist_text}
    Pregunta del usuario: "{pregunta_usuario}"

    Devuelve SOLO la consulta SQL (sin explicaciones).
    """
    
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query_bruta = query_chain.invoke({"question": prompt_con_instrucciones})
        m = re.search(r'(?is)(select\b.+)$', sql_query_bruta.strip()); sql_query_limpia = m.group(1).strip() if m else sql_query_bruta.strip()
        sql_query_limpia = re.sub(r'(?is)^```sql|```$', '', sql_query_limpia).strip(); sql_query_limpia = _asegurar_select_only(sql_query_limpia)
        st.code(sql_query_limpia, language='sql')
        with st.spinner("⏳ Ejecutando consulta..."):
            with db._engine.connect() as conn: df = pd.read_sql(text(sql_query_limpia), conn)
        st.success(f"✅ ¡Consulta ejecutada! Filas: {len(df)}")

        try:
            if not df.empty:
                # Extraer año de la consulta SQL (si existe)
                year_match = re.search(r"YEAR\([^)]*\)\s*=\s*(\d{4})", sql_query_limpia)
                year_value = year_match.group(1) if year_match else None

                # Agregar columna Año solo si se detecta
                if year_value and "Año" not in df.columns:
                    df.insert(0, "Año", year_value)

                # 🔹 Detectar solo columnas numéricas que sí deben sumarse (evitar Mes, Año, Fecha...)
                value_cols = [
                    c for c in df.select_dtypes("number").columns
                    if not re.search(r"(?i)\b(mes|año|dia|fecha)\b", c)
                ]

                # 🔹 Agregar fila Total si hay columnas de valor
                if value_cols:
                    total_row = {col: df[col].sum() if col in value_cols else "" for col in df.columns}
                    total_row[df.columns[0]] = "Total"  # siempre en la primera columna
                    df.loc[len(df)] = total_row

                # 🎨 Estilo visual para la fila Total
                def highlight_total(row):
                    return [
                        "font-weight: bold; background-color: #f8f9fa; border-top: 2px solid #999;"
                        if str(row.iloc[0]).lower() == "total" else ""
                    ] * len(row)

                styled_df = df.style.apply(highlight_total, axis=1)
                return {"sql": sql_query_limpia, "df": df, "styled": styled_df}

        except Exception as e:
            st.warning(f"No se pudo aplicar formato ni totales: {e}")
            
        return {"sql": sql_query_limpia, "df": df}
    except Exception as e:
        st.warning(f"❌ Error en la consulta directa. Intentando método alternativo... Detalle: {e}")
        return {"sql": None, "df": None, "error": str(e)}

def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str, hist_text: str):
    st.info("🤔 Activando el agente SQL experto como plan B.")
    prompt_sql = (f"Tu tarea es responder la pregunta consultando la tabla 'automundial'.\n{hist_text}\nDevuelve ÚNICAMENTE una tabla en formato Markdown (con encabezados). NUNCA resumas ni expliques. El SQL interno NO DEBE CONTENER 'LIMIT' excepción si el cliente lo solicita.\nPregunta: {pregunta_usuario}")
    try:
        with st.spinner("💬 Pidiendo al agente SQL que responda..."):
            res = agente_sql.invoke(prompt_sql)
            texto = res["output"] if isinstance(res, dict) and "output" in res else str(res)
        st.info("📝 Intentando convertir la respuesta en una tabla de datos..."); df_md = markdown_table_to_df(texto)
        if df_md.empty: st.warning("La conversión de Markdown a tabla no produjo filas. Se mostrará la salida cruda.")
        return {"texto": texto, "df": df_md}
    except Exception as e:
        st.error(f"❌ El agente SQL experto también encontró un problema: {e}")
        return {"texto": f"[SQL_ERROR] {e}", "df": pd.DataFrame()}
def analizar_con_datos(pregunta_usuario: str, hist_text: str, df: pd.DataFrame | None, feedback: str = None):
    st.info("\n🧠 El analista experto está examinando los datos...")
    correccion_prompt = ""
    if feedback:
        st.warning(f"⚠️ Reintentando con feedback: {feedback}")
        correccion_prompt = (f'INSTRUCCIÓN DE CORRECCIÓN: Tu respuesta anterior fue incorrecta. Feedback: "{feedback}". Genera una NUEVA respuesta corrigiendo este error.')
    preview = _df_preview(df, 50) or "(sin datos en vista previa; verifica la consulta)"
    prompt_analisis = f"""{correccion_prompt}\nEres IANA, un analista de datos senior EXTREMADAMENTE PRECISO y riguroso.\n---\n<<< REGLAS CRÍTICAS DE PRECISIÓN >>>\n1. **NO ALUCINAR**: NUNCA inventes números, totales, porcentajes o nombres de productos/categorías que no estén EXPRESAMENTE en la tabla de 'Datos'.\n2. **DATOS INCOMPLETOS**: Reporta los vacíos (p.ej., "sin datos para Marzo") sin inventar valores.\n3. **VERIFICAR CÁLCULOS**: Antes de escribir un número, revisa el cálculo (sumas/conteos/promedios) con los datos.\n4. **CITAR DATOS**: Basa CADA afirmación que hagas en los datos visibles en la tabla.\n---\nPregunta Original: {pregunta_usuario}\n{hist_text}\nDatos para tu análisis (usa SÓLO estos):\n{preview}\n---\nFORMATO OBLIGATORIO:\n📌 Análisis Ejecutivo de datos:\n1. Calcular totales y porcentajes clave.\n2. Detectar concentración.\n3. Identificar patrones temporales.\n4. Analizar dispersión.\nEntregar el resultado en 3 bloques:\n📌 Resumen Ejecutivo: hallazgos principales con números.\n🔍 Números de referencia: totales, promedios, ratios.\n⚠ Importante: Sé muy breve, directo y diciente."""
    with st.spinner("💡 Generando análisis avanzado..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis
def responder_conversacion(pregunta_usuario: str, hist_text: str):
    st.info("💬 Activando modo de conversación...")
    prompt_personalidad = f"""Tu nombre es IANA, una IA amable de automundial. Ayuda a analizar datos.\nSi el usuario hace un comentario casual, responde amablemente de forma natural, muy humana y redirígelo a tus capacidades.\n{hist_text}\nPregunta: "{pregunta_usuario}" """
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
        df_preview = _df_preview(res_analisis.get("df"), 50) or "(sin vista previa de datos)"
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

def obtener_datos_sql(pregunta_usuario: str, hist_text: str) -> dict:
    if any(keyword in pregunta_usuario.lower() for keyword in ["anterior", "esos datos", "esa tabla"]):
        for msg in reversed(st.session_state.get('messages', [])):
            if msg.get('role') == 'assistant':
                content = msg.get('content', {}); df_prev = content.get('df')
                if isinstance(df_prev, pd.DataFrame) and not df_prev.empty:
                    st.info("💡 Usando datos de la respuesta anterior para la nueva solicitud.")
                    return {"df": df_prev}
    res_real = ejecutar_sql_real(pregunta_usuario, hist_text)
    if res_real.get("df") is not None and not res_real["df"].empty:
        return res_real
    return ejecutar_sql_en_lenguaje_natural(pregunta_usuario, hist_text)

def orquestador(pregunta_usuario: str, chat_history: list):
    with st.expander("⚙️ Ver Proceso de IANA", expanded=False):
        hist_text = get_history_text(chat_history)
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

        res_datos = obtener_datos_sql(pregunta_usuario, hist_text)
        if res_datos.get("df") is None or res_datos["df"].empty:
            return {"tipo": "error", "texto": "Lo siento, no pude obtener datos para tu pregunta. Intenta reformularla."}

        #if clasificacion == "consulta":
        #    st.success("✅ Consulta directa completada.")
        #    return interpretar_resultado_sql(res_datos)


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
    st.session_state.messages = [{"role": "assistant", "content": {"texto": "¡Hola! Soy IANA, tu asistente de IA de automundial. ¿Qué te gustaría saber?"}}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message.get("content", {});
        if isinstance(content, dict):
            if content.get("texto"): st.markdown(content["texto"])
            if isinstance(content.get("df"), pd.DataFrame) and not content["df"].empty: st.dataframe(content["df"])
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
                if isinstance(res.get("df"), pd.DataFrame) and not res["df"].empty: st.dataframe(res["df"])
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
    

