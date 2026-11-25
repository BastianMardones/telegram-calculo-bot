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
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

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
BASE_SYSTEM_INSTRUCTION = """Eres el Tutor Experto de 'Cálculo Integral' (Código 242150, Semestre 2026-2) de la Universidad del Bío-Bío (UBB), cátedra del **Prof. Dr. Eligio Antonio Colmenares**.

Tienes cargados e indexados los 6 documentos oficiales actuales del curso:
1. **Certamen 3 – Series Infinitas (242150 - Semestre 2025-2 - Prof. Eligio Colmenares, Prioridad Máxima):** Los 6 problemas completos con su solucionario paso a paso:
   • Problema 1: 3 series numéricas (geométrica divergente `r=4/3`, criterio de divergencia `lím=1/4`, y álgebra de series geométrica `3/eⁿ` menos telescópica `2/(5n(n+1))`).
   • Problema 2: Serie telescópica `∑ 3/(n(n+3))` expresando `{sₙ}` explícitamente hasta llegar a suma `11/6`.
   • Problema 3: Comparación por paso al límite para `∑ (2n²+3n)/√(5+n⁵)` comparando con `1/√n` (`L=2`, diverge).
   • Problema 4: 3 series de potencias indicando centro `a`, radio `R` e intervalo `I` con análisis de extremos.
   • Problema 5: Demostración de la suma geométrica `∑ xⁿ = 1/(1-x)` en `(-1,1)` y deducción por integración término a término de `tan⁻¹(x) = ∑ (-1)ⁿ x^(2n+1)/(2n+1)`.
   • Problema 6: Serie de Taylor (Maclaurin) de `f(x) = cos(x)` en `a = 0` por definición de derivadas + `R = +∞, I = ℝ`.
2. **Evaluación Integradora de Cálculo Integral (242150 - Semestre I-2024 - Prof. Eligio Colmenares, Prioridad Máxima):** Los 5 problemas completos con su solucionario verificado paso a paso:
   • Problema 1: 6 integrales (sustitución con raíz + división polinómica; completación de cuadrados con integral impropia Tipo 2 escondida en el extremo; integración por partes doble; integral impropia doble en 0⁺ y +∞; sustitución + trigonométrica; fracciones parciales con factor repetido y cuadrático irreducible).
   • Problema 2: (a) Plantear área entre curvas que se cruzan dentro del intervalo (partir integral en el punto de intersección); (b) Plantear volumen por **secciones transversales conocidas** (cuadrados perpendiculares al eje x, partiendo en la intersección de las parábolas).
   • Problema 3: Sucesión recurrente (`a₁ = 1, aₙ₊₁ = ½(aₙ + 2/aₙ)`): probar cota cuadrática `aₙ² ≥ 2`, monotonía por diferencia, acotamiento inferior por inducción, convergencia monótona y límite `√2`.
   • Problema 4: Series numéricas con el criterio específico solicitado (Comparación por paso al límite, Criterio de la Razón con `(1 + 1/n)ⁿ → e`, y Criterio de la Raíz).
   • Problema 5: Radio e Intervalo de convergencia de Serie de Potencias analizando ambos extremos por separado.
3. **Guía de preparación para el Test - Unidad IV: Sucesiones y Series Infinitas** (`guia_preparacion_test_unidad_IV_series.pdf`): Mapa conceptual oficial y los 48 ejercicios (Secciones A hasta L).
4. **Listado de Cálculo Integral de la Unidad IV: Sucesiones y Series Infinitas 2026-2** (`ListadoUnidadI IV 2026.pdf`): Ejercicios 1 al 22 completos.
5. **Diapositivas oficiales de clases: Sucesiones y Series Infinitas 2026-2** (`SucesionesSeries2026.pdf`, 41 páginas).
6. **Libro / Apuntes oficiales de clase de Cálculo Integral** (`calculo_integral (10).pdf`, 174 páginas del Prof. Eligio Antonio Colmenares).

REGLAS ESTRICTAS DE FORMATO MATEMÁTICO PARA SAMSUNG GALAXY WATCH 4 (396x396 px):
1. PROHIBIDO USAR LATEX CRUDO (PROHIBIDO \\int, \\frac, \\sqrt, \\sum, \\lim, \\left, \\right, $$):
   - En el reloj el código LaTeX NO se renderiza.
   - Escribe TODA expresión matemática exclusivamente con **Unicode matemático limpio**:
     • Series y sumas: `∑_{n=1}^∞ aₙ`, suma parcial `sₙ`
     • Límites: `lím_{n→∞} aₙ = 0`
     • Integrales y raíces: `∫`, `√(n² + 1)`, `³√(n³ + 3n)`
     • Fracciones en una línea clara: `(2n + 1) / (3n + 4)`
     • Potencias y subíndices Unicode: `x²`, `x³`, `xⁿ`, `aₙ`, `bₙ`, `sₙ`
     • Símbolos: `π`, `∞`, `θ`, `≈`, `≤`, `≥`, `≠`, `⇒`, `→`
     • Funciones en español: `sen(x)`, `cos(x)`, `tan(x)`, `ln(n)`, `arctan(n)`

2. ESTRUCTURA OBLIGATORIA DE RESPUESTA (DISEÑADA PARA LEER EN LA MUÑECA SIN SALTARSE PASOS):
   - Cero saludos ni introducciones de relleno.
   - Comienza SIEMPRE con este bloque de 2 líneas arriba del todo:
     🎯 **Conclusión / Resultado:** `[Ej: DIVERGE / CONVERGE (Suma = 11/6) / R = 4, I = (-1, 7]]`
     🔑 **Criterio / Idea clave:** `[Ej: Criterio de la Razón (L = 4/3 > 1) / Serie telescópica / Fracciones parciales]`
   - Luego muestra **📝 Desarrollo completo paso a paso (tal como se escribe en hoja de certamen):** con líneas cortas y verticales para el reloj, pero **100% COMPLETO SIN OMITIR NINGÚN PASO ALGEBRAICO NI NOTACIÓN FORMAL**.

3. REGLA DE ORO DE RIGOR FORMAL (NUNCA OMITIR `lím` NI `|...|` NI PASOS ALGEBRAICOS):
   - **PROHIBIDO perder `lím_{n→∞}` o las barras de valor absoluto `| ... |` a mitad del desarrollo:**
     • En Criterio de la Razón (`L = lím_{n→∞} |aₙ₊₁ / aₙ|`), Criterio de la Raíz (`L = lím_{n→∞} ⁿ√|aₙ|`), Comparación en el límite (`L = lím_{n→∞} (aₙ / bₙ)`), Criterio de Divergencia (`lím_{n→∞} aₙ`) o integrales impropias (`lím_{b→∞}`), **DEBE aparecer `lím_{n→∞}` (y `| ... |` cuando el criterio lo lleva) en TODAS Y CADA UNA de las líneas de la cadena de igualdades** hasta el último paso en que se evalúa el límite.
     • NUNCA calcules `aₙ₊₁ / aₙ` aparte sin `lím` ni `|...|` para ponerlos recién al final.
   - **PROHIBIDO saltarse álgebra intermedia:**
     • Al escribir `aₙ₊₁`, muestra primero la sustitución literal de `n` por `(n + 1)` y el cálculo explícito del nuevo exponente o factor (ej: `2(n+1) − 1 = 2n + 2 − 1 = 2n + 1`).
     • Muestra la división de fracciones convertida en producto, las cancelaciones de factores, la resta explícita de exponentes con paréntesis (ej: `(2n+1) − (2n−1) = 2n + 1 − 2n + 1 = 2`), y el paso previo al resultado final (ej: `= lím_{n→∞} |2² / 3| = lím_{n→∞} (4/3) = 4/3`).
   - **Ejemplo modelo obligatorio de cadena formal (seguir SIEMPRE este rigor):**
     • `aₙ = (6 · 2^(2n−1)) / 3ⁿ`
     • `aₙ₊₁ = (6 · 2^(2(n+1)−1)) / 3^(n+1)`
       (Cálculo auxiliar: `2(n+1) − 1 = 2n + 2 − 1 = 2n + 1`)
       ⇒ `aₙ₊₁ = (6 · 2^(2n+1)) / 3^(n+1)`
     • `L = lím_{n→∞} | aₙ₊₁ / aₙ |`
     • `= lím_{n→∞} | [ (6 · 2^(2n+1)) / 3^(n+1) ] / [ (6 · 2^(2n−1)) / 3ⁿ ] |`
     • `= lím_{n→∞} | (6 · 2^(2n+1) · 3ⁿ) / (6 · 2^(2n−1) · 3^(n+1)) |`
     • `= lím_{n→∞} | 2^((2n+1) − (2n−1)) / 3^((n+1) − n) |`
       (Cálculo auxiliar: `(2n+1) − (2n−1) = 2n + 1 − 2n + 1 = 2`)
     • `= lím_{n→∞} | 2² / 3 |`
     • `= lím_{n→∞} (4 / 3) = 4 / 3`
     • Como `L = 4/3 > 1`, por el Criterio de la Razón la serie **DIVERGE**.

4. RIGOR CONCEPTUAL DE LA CÁTEDRA Y EVALUACIÓN INTEGRADORA (PROF. ELIGIO COLMENARES):
   - **Integrales impropias escondidas:** Revisa SIEMPRE si el denominador se anula en algún extremo o punto interior (como `∫₂⁵ dx / √(5 + 4x - x²)` en `x = 5` o `∫₀^∞ e^(-√x)/√x dx` en `0⁺` y `+∞`) y resuélvelas formalmente manteniendo `lím_{b→5⁻}` en cada igualdad hasta evaluar.
   - **Áreas y Volúmenes por Secciones Transversales:** Verifica SIEMPRE si las curvas se cruzan dentro del intervalo de integración (si se cruzan, parte la integral en el cruce). En sólidos con secciones transversales perpendiculares al eje `x` (cuadrados, semicírculos, triángulos), integra respecto a `x` usando `V = ∫ A(x) dx` y separa la integral si cambia la curva frontera.
   - **Sucesiones recurrentes:** Demuestra cotas, monotonía (`aₙ₊₁ - aₙ`), acotamiento por inducción, invoca explícitamente el **Teorema de la Convergencia Monótona** y luego pasa al límite en la recurrencia descartando raíces fuera del rango.
   - **Series y Series de Potencias:** Distingue con precisión entre `{aₙ}`, `{sₙ}` y `∑ aₙ` (`lím aₙ = 0` es necesaria pero NO suficiente). Al hallar el intervalo de convergencia de una serie de potencias, analiza SIEMPRE los dos extremos por separado paso a paso.
"""

SUPERSCRIPTS = str.maketrans("0123456789+-=n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼ⁿ")
SUBSCRIPTS = str.maketrans("0123456789+-=nij", "₀₁₂₃₄₅₆₇₈₉₊₋₌ₙᵢⱼ")

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
        (r"\\lvert|\\rvert", "|"),
        (r"\\left|\\right", ""),
    ]
    for patron, sub in reemplazos:
        texto = re.sub(patron, sub, texto)

    texto = re.sub(r"\^\{∞\}", "^∞", texto)
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
    relevant_docs = SEARCH_ENGINE.search(message_text, top_k=6)
    system_prompt = (
        f"{BASE_SYSTEM_INSTRUCTION}\n\n"
        "=== MATERIAL OFICIAL DEL PROF. ELIGIO COLMENARES RECUPERADO ===\n"
        f"{relevant_docs}\n"
        "=== FIN DE MATERIAL RECUPERADO ==="
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
            "max_tokens": 8192 if use_reasoner else 4096,
        }
        if not use_reasoner:
            payload["temperature"] = 0.1

        resp = requests.post(DEEPSEEK_BASE_URL, headers=headers, json=payload, timeout=120)
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
        "∑ <b>Cálculo Integral UBB (Prof. Eligio Colmenares + Watch 4)</b>\n\n"
        "📚 <b>6 Documentos Oficiales Cargados:</b>\n"
        "• <code>Certamen 3 – Series Infinitas (2025-2)</code> (Series, Telescópica, Potencias, arctan, Taylor cos)\n"
        "• <code>Evaluación Integradora (I-2024)</code> (Integrales, Áreas, Secciones, Recurrencia, Series)\n"
        "• <code>Guía Preparación Test Unidad IV (Series)</code> (Ej. 1–48)\n"
        "• <code>Listado Unidad IV 2026 (Sucesiones y Series)</code> (Ej. 1–22)\n"
        "• <code>SucesionesSeries2026.pdf</code> (41 slides) + <code>Apuntes (174 págs)</code>\n\n"
        "⚡ <b>Comandos rápidos para tu reloj:</b>\n"
        "• <code>/c3 [N°]</code> - Resolver/practicar Certamen 3 Series Infinitas (ej: <code>/c3 2</code>, <code>/c3 4c</code>, <code>/c3 5</code>, <code>/c3 6</code>)\n"
        "• <code>/integradora [N°]</code> - Resolver/practicar Evaluación Integradora (ej: <code>/integradora 1f</code>, <code>/integradora 2b</code>)\n"
        "• <code>/test [N°]</code> - Resolver ej. 1 al 48 de la Guía del Test de Series\n"
        "• <code>/listado [N°]</code> - Resolver ej. 1 al 22 del Listado Unidad IV\n"
        "• <code>/series</code> - Mapa conceptual y criterios de convergencia\n"
        "• <code>/practicar [tema]</code> - Pregunta tipo Certamen 3 / Integradora\n"
        "• <code>/nuevo</code> - Reiniciar conversación"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.HTML)

async def nuevo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    await update.message.reply_text("🔄 Memoria reiniciada. ¿Qué sucesión, serie o ejercicio vemos?")

async def c3_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    num = " ".join(context.args) if context.args else ""
    if num:
        prompt = (
            f"Muéstrame el enunciado y la solución completa paso a paso (sin omitir lím, |...| ni ningún paso algebraico intermedio) "
            f"del Problema '{num}' del Certamen 3 – Series Infinitas (Semestre 2025-2 - Prof. Eligio Colmenares)."
        )
    else:
        prompt = (
            "Haz un resumen compacto para mi Galaxy Watch 4 con los 6 problemas del Certamen 3 – Series Infinitas (2025-2, Prof. Eligio Colmenares): "
            "1. 3 series numéricas (geométrica, divergencia, álgebra de series), 2. Telescópica 3/(n(n+3)) por {sₙ}, "
            "3. Comparación en el límite con 1/√n, 4. 3 series de potencias (centro, radio e intervalo), "
            "5. Demostración de 1/(1-x) y tan⁻¹(x), 6. Serie de Taylor de cos(x) en a=0 + R e I, "
            "y dime que puedo escribir /c3 [1a..6] para ver cualquier solución paso a paso."
        )
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        reply = await asyncio.to_thread(ask_deepseek, user_id, prompt, bool(num))
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def integradora_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    num = " ".join(context.args) if context.args else ""
    if num:
        prompt = (
            f"Muéstrame el enunciado y la solución completa paso a paso (sin omitir lím, |...| ni ningún paso algebraico intermedio) "
            f"del Problema '{num}' de la Evaluación Integradora de Cálculo Integral (242150 - Prof. Eligio Colmenares)."
        )
    else:
        prompt = (
            "Haz un resumen compacto para mi Galaxy Watch 4 con los 5 problemas de la Evaluación Integradora de Cálculo Integral (Prof. Eligio Colmenares): "
            "1. Las 6 integrales (1a–1f, incluyendo impropias y fracciones parciales), 2. Área con cruce y Volumen por secciones transversales cuadradas, "
            "3. Sucesión recurrente aₙ₊₁=½(aₙ+2/aₙ), 4. Series por Comparación límite, Razón y Raíz, y 5. Radio e intervalo de convergencia, "
            "y dime que puedo escribir /integradora [1a..5] o /practicar integradora."
        )
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        reply = await asyncio.to_thread(ask_deepseek, user_id, prompt, bool(num))
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    num = " ".join(context.args) if context.args else ""
    if num:
        prompt = (
            f"Muéstrame el enunciado y la solución completa paso a paso (sin omitir lím, |...| ni ningún paso algebraico intermedio) "
            f"del ejercicio '{num}' de la Guía de preparación para el test de la Unidad IV (Sucesiones y Series Infinitas)."
        )
    else:
        prompt = (
            "Haz un resumen ultra compacto para mi Galaxy Watch 4 de las secciones A hasta L (ejercicios 1 al 48) "
            "de la Guía de preparación para el test de la Unidad IV (Series Infinitas) del Prof. Eligio Colmenares, "
            "y dime que puedo escribir /test [número] (ej: /test 1, /test 15, /test 48) para ver cualquier ejercicio resuelto."
        )
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        reply = await asyncio.to_thread(ask_deepseek, user_id, prompt, bool(num))
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def listado_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    num = " ".join(context.args) if context.args else ""
    if num:
        prompt = (
            f"Muéstrame el enunciado y la resolución completa paso a paso (sin omitir lím, |...| ni ningún paso algebraico intermedio) "
            f"del ejercicio '{num}' del Listado de Cálculo Integral de la Unidad IV (Sucesiones y Series Infinitas 2026-2)."
        )
    else:
        prompt = (
            "Haz un resumen compacto para mi Galaxy Watch 4 de los 22 ejercicios del Listado de la Unidad IV "
            "(Sucesiones: ejercicios 1 al 13; Series Infinitas y de Potencias: ejercicios 12 al 22) del Prof. Eligio Colmenares, "
            "y dime que puedo escribir /listado [número] (ej: /listado 5, /listado 14, /listado 22) para resolver cualquiera."
        )
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        reply = await asyncio.to_thread(ask_deepseek, user_id, prompt, bool(num))
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def series_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prompt = (
        "Muéstrame el Mapa Conceptual Oficial de Series Infinitas de la Unidad IV del Prof. Eligio Colmenares "
        "en formato vertical ultra compacto para mi Galaxy Watch 4 (Convergencia por sₙ, Criterio de divergencia, "
        "Geométricas, Series p, Telescópicas, Comparación directa, Comparación por límite, Leibniz y Convergencia absoluta)."
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
    tema = " ".join(context.args) if context.args else "Evaluación Integradora (técnicas de integración, impropias, áreas/secciones transversales, sucesiones recurrentes o series)"
    prompt = (
        f"Propón un ejercicio estilo Evaluación Integradora del Prof. Eligio Colmenares (UBB) sobre: '{tema}'. "
        "Escríbelo en Unicode limpio para mi Galaxy Watch 4 y espera a que te diga cómo lo resolvería."
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
    caption = update.message.caption or "Resuelve este ejercicio de Cálculo Integral paso a paso completo (sin omitir lím, |...| ni ningún paso algebraico intermedio)."
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
    app.add_handler(CommandHandler("c3", c3_cmd))
    app.add_handler(CommandHandler("certamen3", c3_cmd))
    app.add_handler(CommandHandler("integradora", integradora_cmd))
    app.add_handler(CommandHandler("int", integradora_cmd))
    app.add_handler(CommandHandler("test", test_cmd))
    app.add_handler(CommandHandler("listado", listado_cmd))
    app.add_handler(CommandHandler("series", series_cmd))
    app.add_handler(CommandHandler("c2", test_cmd))
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
