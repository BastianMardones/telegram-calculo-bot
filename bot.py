# -*- coding: utf-8 -*-
import os
import io
import re
import json
import math
import html
import sys
import time
import logging
import threading
import asyncio
from collections import Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import requests
from dotenv import load_dotenv
from PIL import Image
from pypdf import PdfReader
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

try:
    import google.generativeai as genai
except ImportError:
    genai = None

load_dotenv()

def clean_env_key(var_name: str) -> str:
    val = (os.getenv(var_name) or "").strip()
    if "=" in val and not val.startswith("sk-"):
        val = val.split("=", 1)[-1].strip()
    return val.strip('"').strip("'").strip("`").replace("\r", "").replace("\n", "").strip()

DEEPSEEK_API_KEY = (
    clean_env_key("DEEPSEEK_API_KEY")
    or clean_env_key("DEEPSEEK_KEY")
    or clean_env_key("OPENAI_API_KEY")
)
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")

GEMINI_API_KEY = (
    clean_env_key("GEMINI_API_KEY")
    or clean_env_key("GOOGLE_API_KEY")
    or clean_env_key("GEMINI_KEY")
)

TELEGRAM_BOT_TOKEN = clean_env_key("TELEGRAM_BOT_TOKEN")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "documentos"
INDEX_FILE = BASE_DIR / "documentos_index.json"
DOCS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY, transport="rest")
    except Exception:
        genai.configure(api_key=GEMINI_API_KEY)

# ============================================================
# SERVIDOR WEB KEEP-ALIVE PARA RENDER (24/7 SIN PC)
# ============================================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        ds_status = (
            f"{DEEPSEEK_API_KEY[:5]}...{DEEPSEEK_API_KEY[-4:]}"
            if len(DEEPSEEK_API_KEY) > 10
            else ("OK" if DEEPSEEK_API_KEY else "FALTANTE")
        )
        msg = (
            "=== Bot Calculo Integral Stewart (Watch 4 + DeepSeek) ===\n"
            f"Telegram Token: {'OK' if TELEGRAM_BOT_TOKEN else 'FALTANTE'}\n"
            f"DeepSeek API Key: {ds_status}\n"
            f"Documentos indexados: {len(SEARCH_ENGINE.chunks)} fragmentos\n"
            "Estado: En ejecucion 24/7\n"
        )
        self.wfile.write(msg.encode("utf-8"))

    def log_message(self, format, *args):
        pass

def run_health_check_server():
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

def run_keep_alive_pinger():
    """Hace ping cada 10 minutos a RENDER_EXTERNAL_URL para evitar que Render duerma el bot tras 15 min."""
    time.sleep(30)
    while True:
        url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("KEEP_ALIVE_URL")
        if url:
            try:
                requests.get(url, timeout=10)
                logger.info(f"Keep-alive ping enviado a {url}")
            except Exception as e:
                logger.warning(f"Keep-alive ping fallo: {e}")
        time.sleep(600)

# ============================================================
# BUSCADOR INTELIGENTE BM25 PARA GUÍAS Y COMPENDIO STEWART
# ============================================================
STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "de", "del", "al", "a", "en", "con",
    "por", "para", "y", "o", "que", "como", "es", "son", "se", "su", "lo", "mas"
}

def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9_]{2,}", text.lower())
    return [w for w in words if w not in STOPWORDS]

class SmartDocumentIndex:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.chunks = []
        self.df = Counter()
        self.avg_len = 1.0
        self.total_docs = 0
        self.reload()

    def reload(self):
        self.chunks = []
        self.df = Counter()
        if not self.index_path.exists():
            return
        try:
            docs = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.total_docs = len(docs)
            for doc in docs:
                title = doc.get("title", "Stewart")
                category = doc.get("category", "CALCULO")
                priority = float(doc.get("priority", 2))
                content = doc.get("content", "")
                paragraphs = content.split("\n\n")
                cur_text, cur_len = [], 0
                for p in paragraphs:
                    if cur_len + len(p) > 1600 and cur_text:
                        self._add_chunk(title, category, priority, "\n\n".join(cur_text))
                        cur_text, cur_len = [p], len(p)
                    else:
                        cur_text.append(p)
                        cur_len += len(p)
                if cur_text:
                    self._add_chunk(title, category, priority, "\n\n".join(cur_text))
            if self.chunks:
                self.avg_len = sum(c["len"] for c in self.chunks) / len(self.chunks)
        except Exception as e:
            logger.error(f"Error cargando índice: {e}")

    def _add_chunk(self, title: str, category: str, priority: float, text: str):
        tokens = tokenize(f"{title} {text}")
        if not tokens:
            return
        tf = Counter(tokens)
        for w in set(tokens):
            self.df[w] += 1
        self.chunks.append({
            "title": title,
            "category": category,
            "priority": priority,
            "text": text,
            "tf": tf,
            "len": len(tokens),
        })

    def add_new_document(self, title: str, content: str):
        docs = []
        if self.index_path.exists():
            try:
                docs = json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception:
                docs = []
        docs.insert(0, {
            "title": title,
            "path": f"upload/{title}",
            "category": "GUIA_O_CERTAMEN",
            "priority": 4,
            "content": content,
        })
        self.index_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
        self.reload()

    def search(self, query: str, top_k: int = 4) -> str:
        if not self.chunks:
            return ""
        q_tokens = tokenize(query)
        if not q_tokens:
            selected = self.chunks[:3]
        else:
            N = len(self.chunks)
            scored = []
            for idx, ch in enumerate(self.chunks):
                score = 0.0
                for qt in q_tokens:
                    if qt in ch["tf"]:
                        df_w = self.df.get(qt, 0)
                        idf = math.log(1.0 + (N - df_w + 0.5) / (df_w + 0.5))
                        f = ch["tf"][qt]
                        denom = f + 1.5 * (0.25 + 0.75 * (ch["len"] / self.avg_len))
                        score += idf * ((f * 2.5) / denom)
                if score > 0:
                    score *= (ch["priority"] / 2.0)
                    scored.append((score, idx))
            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [self.chunks[i] for _, i in scored[:top_k]]
            if not selected:
                selected = self.chunks[:2]
        return "\n\n---\n\n".join(f"[{c['title']}]\n{c['text']}" for c in selected)

SEARCH_ENGINE = SmartDocumentIndex(INDEX_FILE)
threading.Thread(target=run_health_check_server, daemon=True).start()
threading.Thread(target=run_keep_alive_pinger, daemon=True).start()

# ============================================================
# PROMPT SISTEMA: CÁLCULO INTEGRAL STEWART (FORMATO WATCH 4)
# ============================================================
BASE_SYSTEM_INSTRUCTION = """Eres el Tutor Experto de 'Cálculo Integral' (basado en Cálculo de Una Variable de James Stewart y certámenes universitarios de Ingeniería UBB).

REGLAS ESTRICTAS DE FORMATO MATEMÁTICO PARA SAMSUNG GALAXY WATCH 4 (396x396 px):
1. PROHIBIDO USAR LATEX CRUDO (PROHIBIDO \\int, \\frac, \\sqrt, \\left, \\right, $$):
   - En el reloj el código LaTeX NO se renderiza.
   - Escribe TODA expresión matemática exclusivamente con **Unicode matemático limpio**:
     • Símbolo de integral: `∫` (ej: `∫₀^π`, `∫ (x² + 1) dx`)
     • Raíces: `√(x² - 4)`, `³√(x)`
     • Fracciones en una sola línea clara: `(x² + 1) / (x - 2)` o `½`, `⅓`, `¼`
     • Potencias y subíndices Unicode: `x²`, `x³`, `x⁴`, `xⁿ`, `u₁`, `u₂`, `aₙ`
     • Símbolos: `π`, `∞`, `θ`, `α`, `β`, `∑`, `lím_{x→∞}`, `≈`, `≤`, `≥`, `≠`, `⇒`, `→`
     • Funciones en español: `sen(x)`, `cos(x)`, `tan(x)`, `sec(x)`, `ln|x|`, `arcsen(x)`, `arctan(x)`

2. ESTRUCTURA OBLIGATORIA DE RESPUESTA (DISEÑADA PARA LEER EN LA MUÑECA):
   - Cero saludos ni texto de relleno.
   - Comienza SIEMPRE con este bloque de 2 líneas arriba del todo:
     🎯 **Resultado:** `[Resultado final exacto + C o valor numérico]`
     🔑 **Método:** `[Ej: Sustitución trigonométrica x = 2·sen(θ) / Partes ILATE / Casquillos]`
   - Luego muestra **📝 Pasos cortos:** numerados (`Paso 1`, `Paso 2`, `Paso 3`...), donde cada línea matemática sea corta y vertical para no hacer scroll horizontal en la pantalla redonda del reloj.

3. RIGOR EN APLICACIONES Y ESTILO DEL CERTAMEN II UBB (PROF. DR. ELIGIO COLMENARES):
   - Tienes cargada la Pauta Oficial del **Certamen II de Cálculo Integral (242150, Semestre 2026-2, Prof. Dr. Eligio Colmenares)**:
     • **Parte I (Demostraciones y TFC):** Ecuaciones integrales derivando con TFC Parte 1 + regla del producto (`[x·f(x)]' = f(x) + x·f'(x)`) + integración por partes y cálculo de la constante con condición inicial `f(1)`; Desigualdad de Cauchy-Schwarz integral.
     • **Parte II (Integrales Definidas):**
       - En **cambio de variable**, si se pide transformar límites, cambia `x∈[a,b]` a `u∈[u(a),u(b)]` y evalúa directo en `u` sin volver a `x`.
       - En **sustitución trigonométrica**, resuelve primero la antiderivada indefinida en `θ`, regresa a la variable original `x` mediante el triángulo rectángulo y luego evalúa con el TFC.
     • **Parte III (Aplicaciones Geométricas - Áreas y Volúmenes):**
       - Interpretación geométrica de la integral definida descomponiendo en áreas conocidas (semicírculo `x²+(y-b)²=r²` + rectángulos) sin calcular antiderivadas.
       - Regiones Tipo I (vertical `dx`) y Tipo II (horizontal `dy`: `∫ [x_der(y) - x_izq(y)] dy`).
       - Volúmenes de revolución por **Discos** (`V = π ∫ R²`), **Arandelas** (`V = π ∫ (R_ext² - r_int²)`) y **Casquillos** (`V = 2π ∫ r·h`), indicando siempre variable de integración, radio exterior e interior.

"""

SUPERSCRIPTS = str.maketrans("0123456789+-=n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼ⁿ")
SUBSCRIPTS = str.maketrans("0123456789+-=nij", "₀₁₂₃₄₅₆₇₈₉⁺⁻⁼ₙᵢⱼ")

def limpiar_latex_a_unicode_watch4(texto: str) -> str:
    # Convertir \frac{a}{b} a (a)/(b)
    for _ in range(3):
        texto = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", texto)
    texto = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", texto)

    reemplazos = [
        (r"\\int", "∫"),
        (r"\\sum", "∑"),
        (r"\\lim", "lím"),
        (r"\\pi", "π"),
        (r"\\infty", "∞"),
        (r"\\theta", "θ"),
        (r"\\alpha", "α"),
        (r"\\beta", "β"),
        (r"\\Delta", "Δ"),
        (r"\\rightarrow|\\to", "→"),
        (r"\\Rightarrow", "⇒"),
        (r"\\le(?:q)?", "≤"),
        (r"\\ge(?:q)?", "≥"),
        (r"\\neq", "≠"),
        (r"\\pm", "±"),
        (r"\\cdot", "·"),
        (r"\\times", "×"),
        (r"\\approx", "≈"),
        (r"\\sin", "sen"),
        (r"\\arcsin", "arcsen"),
        (r"\\left|\\right", ""),
    ]
    for patron, sub in reemplazos:
        texto = re.sub(patron, sub, texto)

    texto = re.sub(r"\^\{([0-9+\-=n]+)\}", lambda m: m.group(1).translate(SUPERSCRIPTS), texto)
    texto = re.sub(r"\^([0-9n])", lambda m: m.group(1).translate(SUPERSCRIPTS), texto)
    texto = re.sub(r"_\{([0-9+\-=nij]+)\}", lambda m: m.group(1).translate(SUBSCRIPTS), texto)
    texto = re.sub(r"_([0-9nij])", lambda m: m.group(1).translate(SUBSCRIPTS), texto)
    texto = re.sub(r"\$\$([^\$]+)\$\$", r"\1", texto)
    texto = re.sub(r"\$([^\$]+)\$", r"\1", texto)
    return texto

def formatear_para_telegram(texto: str) -> str:
    texto = limpiar_latex_a_unicode_watch4(texto)

    bloques_codigo = []
    def proteger_bloque(match):
        bloques_codigo.append(match.group(0))
        return f"__BLOQUE_CODIGO_{len(bloques_codigo)-1}__"

    texto = re.sub(r"```[\s\S]*?```", proteger_bloque, texto)

    lineas_codigo = []
    def proteger_linea(match):
        lineas_codigo.append(match.group(1))
        return f"__LINEA_CODIGO_{len(lineas_codigo)-1}__"

    texto = re.sub(r"`([^`\n]+)`", proteger_linea, texto)
    texto = html.escape(texto)

    texto = re.sub(r"^(#{1,4})\s*(.+)$", r"<b>\2</b>", texto, flags=re.MULTILINE)
    texto = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", texto)
    texto = re.sub(r"\*(.+?)\*", r"<i>\1</i>", texto)

    for i, cod in enumerate(lineas_codigo):
        texto = texto.replace(f"__LINEA_CODIGO_{i}__", f"<code>{html.escape(cod)}</code>")

    for i, b in enumerate(bloques_codigo):
        m = re.match(r"```([a-zA-Z0-9_-]*)\n?([\s\S]*?)```", b)
        if m:
            cod_esc = html.escape(m.group(2))
            texto = texto.replace(f"__BLOQUE_CODIGO_{i}__", f"<pre><code>{cod_esc}</code></pre>")
        else:
            texto = texto.replace(f"__BLOQUE_CODIGO_{i}__", f"<pre><code>{html.escape(b)}</code></pre>")

    return texto

def dividir_en_chunks_markdown(texto: str, max_len: int = 3600) -> list[str]:
    if len(texto) <= max_len:
        return [texto]
    parrafos = texto.split("\n\n")
    chunks, chunk_actual, len_actual = [], [], 0
    for p in parrafos:
        if len_actual + len(p) + 2 > max_len and chunk_actual:
            chunks.append("\n\n".join(chunk_actual))
            chunk_actual, len_actual = [p], len(p)
        else:
            chunk_actual.append(p)
            len_actual += len(p) + 2
    if chunk_actual:
        chunks.append("\n\n".join(chunk_actual))
    return chunks

user_histories = {}

def ask_deepseek(user_id: int, message_text: str, use_reasoner: bool = False) -> str:
    relevant_docs = SEARCH_ENGINE.search(message_text, top_k=3)
    system_prompt = (
        f"{BASE_SYSTEM_INSTRUCTION}\n\n"
        "=== REFERENCIA STEWART / GUÍAS CARGADAS ===\n"
        f"{relevant_docs}\n"
        "=== FIN DE REFERENCIA ==="
    )

    if user_id not in user_histories:
        user_histories[user_id] = []
    if len(user_histories[user_id]) > 6:
        user_histories[user_id] = user_histories[user_id][-6:]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(user_histories[user_id])
    messages.append({"role": "user", "content": message_text})

    model_name = "deepseek-reasoner" if use_reasoner else "deepseek-chat"

    if DEEPSEEK_API_KEY:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 2500,
        }
        if not use_reasoner:
            payload["temperature"] = 0.1

        resp = requests.post(DEEPSEEK_BASE_URL, headers=headers, json=payload, timeout=90)
        if resp.status_code != 200:
            raise RuntimeError(f"DeepSeek API ({resp.status_code}): {resp.text[:300]}")
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        user_histories[user_id].append({"role": "user", "content": message_text})
        user_histories[user_id].append({"role": "assistant", "content": reply})
        return reply

    if GEMINI_API_KEY and genai is not None:
        model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_prompt)
        res = model.generate_content(message_text)
        return res.text

    raise RuntimeError("Configura tu DEEPSEEK_API_KEY en el archivo .env o en Render.")

async def split_and_send(update: Update, text: str):
    for chunk in dividir_en_chunks_markdown(text, max_len=3600):
        chunk_html = formatear_para_telegram(chunk)
        try:
            await update.message.reply_text(chunk_html, parse_mode=ParseMode.HTML)
        except Exception:
            texto_plano = html.unescape(re.sub(r"<[^>]+>", "", chunk_html))
            await update.message.reply_text(texto_plano)

# ============================================================
# COMANDOS DEL BOT DE CÁLCULO INTEGRAL (WATCH 4)
# ============================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    welcome = (
        "∫ <b>Cálculo Integral Tutor (James Stewart + Watch 4)</b>\n\n"
        "⚡ <b>Comandos diseñados para tu reloj:</b>\n"
        "• <code>/c2</code> - Ver/practicar Pauta Certamen II 2026-2 (Prof. Eligio Colmenares)\n"
        "• Escribe cualquier integral, área, volumen o demostración TFC\n"
        "• <code>/r1 [ejercicio]</code> - Resolver con <b>DeepSeek-R1</b> (razonador matemático)\n"
        "• <code>/formulario [tema]</code> - Ver fórmulas (partes, trig, volúmenes, series)\n"
        "• <code>/practicar [tema]</code> - Ejercicio tipo Certamen II UBB\n"
        "• <code>/nuevo</code> - Reiniciar conversación"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.HTML)

async def nuevo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    await update.message.reply_text("🔄 Memoria reiniciada. Envíame tu integral o ejercicio.")

async def certamen2_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    num = " ".join(context.args) if context.args else ""
    if num:
        prompt = (
            f"Muéstrame el enunciado y la solución oficial paso a paso en formato compacto para mi Galaxy Watch 4 "
            f"del Problema '{num}' del Certamen II 2026-2 del Prof. Dr. Eligio Colmenares."
        )
    else:
        prompt = (
            "Haz un resumen compacto para mi Galaxy Watch 4 con los 7 problemas que entraron en el "
            "Certamen II de Cálculo Integral 2026-2 del Prof. Dr. Eligio Colmenares (Parte I: TFC y Cauchy-Schwarz, "
            "Parte II: las 5 integrales definidas, Parte III: área geométrica, región Tipo II y volúmenes de revolución) "
            "y dime que puedo escribir /c2 [número de problema] para ver su solución paso a paso o practicar uno similar."
        )
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        reply = await asyncio.to_thread(ask_deepseek, user_id, prompt, False)
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def formulario_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    tema = " ".join(context.args) if context.args else "métodos de integración, áreas y volúmenes de revolución"
    prompt = (
        f"Muéstrame un formulario ultra compacto en Unicode para leer en mi Galaxy Watch 4 sobre: '{tema}'."
    )
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        reply = await asyncio.to_thread(ask_deepseek, user_id, prompt, False)
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def practicar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    tema = " ".join(context.args) if context.args else "integración o volúmenes de sólidos de revolución"
    prompt = (
        f"Propón un ejercicio nivel certamen universitario (estilo James Stewart) sobre: '{tema}'. "
        "Escríbelo en Unicode limpio y espera a que te diga cómo lo plantearía."
    )
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        reply = await asyncio.to_thread(ask_deepseek, user_id, prompt, False)
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def r1_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    q = " ".join(context.args)
    if not q:
        await update.message.reply_text("💡 Uso: <code>/r1 [integral o problema]</code>", parse_mode=ParseMode.HTML)
        return
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        reply = await asyncio.to_thread(ask_deepseek, user_id, q, True)
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        reply = await asyncio.to_thread(ask_deepseek, user_id, user_text, True)
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    caption = update.message.caption or "Resuelve este ejercicio de Cálculo Integral paso a paso en formato compacto para mi reloj."
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        if not GEMINI_API_KEY or genai is None:
            await update.message.reply_text(
                "⚠️ Para leer fotos de ejercicios, agrega una clave gratuita <code>GEMINI_API_KEY</code> en tus variables de entorno (para OCR) o escribe la integral por texto.",
                parse_mode=ParseMode.HTML,
            )
            return
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
        image.thumbnail((1024, 1024))

        ocr_model = genai.GenerativeModel("gemini-2.5-flash")
        ocr_res = await asyncio.to_thread(
            ocr_model.generate_content,
            ["Transcribe con total precisión todas las integrales, límites, funciones y enunciados matemáticos de esta imagen:", image],
        )
        full_query = f"[Ejercicio matemático extraído de foto]:\n{ocr_res.text}\n\nInstrucción: {caption}"
        reply = await asyncio.to_thread(ask_deepseek, user_id, full_query, True)
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ Error en foto: {e}")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    filename = doc.file_name
    ext = Path(filename).suffix.lower()
    if ext not in [".pdf", ".txt", ".md"]:
        await update.message.reply_text("⚠️ Envía un archivo .pdf, .txt o .md.")
        return

    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        dest_path = DOCS_DIR / filename
        doc_file = await doc.get_file()
        await doc_file.download_to_drive(dest_path)

        if ext == ".pdf":
            r = PdfReader(str(dest_path))
            contenido = "\n\n".join(f"[Pág {i+1}]\n{p.extract_text() or ''}" for i, p in enumerate(r.pages))
        else:
            contenido = dest_path.read_text(encoding="utf-8", errors="ignore")

        if contenido.strip():
            SEARCH_ENGINE.add_new_document(filename, contenido)
            await update.message.reply_text(
                f"✅ <b>{html.escape(filename)}</b> incorporado al buscador de Cálculo Integral.",
                parse_mode=ParseMode.HTML,
            )
    except Exception as e:
        await update.message.reply_text(f"❌ Error al indexar: {e}")

def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("Falta TELEGRAM_BOT_TOKEN en el archivo .env o variables de entorno.")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ayuda", start))
    app.add_handler(CommandHandler("nuevo", nuevo))
    app.add_handler(CommandHandler("c2", certamen2_cmd))
    app.add_handler(CommandHandler("certamen2", certamen2_cmd))
    app.add_handler(CommandHandler("formulario", formulario_cmd))
    app.add_handler(CommandHandler("practicar", practicar_cmd))
    app.add_handler(CommandHandler("r1", r1_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("[+] Bot Cálculo Integral (Stewart + Watch 4) iniciado.")
    app.run_polling()

if __name__ == "__main__":
    main()
