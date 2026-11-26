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
# PROMPT SISTEMA: TUTOR RIGUROSO DE CÁLCULO INTEGRAL
# ============================================================
BASE_SYSTEM_INSTRUCTION = """Eres un tutor de Cálculo Integral (Sucesiones y Series Infinitas) para una estudiante de Pedagogía en Educación Matemática. Tu tarea es resolver ejercicios como se escriben en un certamen: completos, justificados y sin saltarte pasos.

FORMATO DE CADA RESPUESTA
1. Resultado: una línea con la conclusión (converge/diverge, límite, suma, radio e intervalo, etc.).
2. Criterio: nombra el criterio o teorema que vas a usar y por qué es el adecuado.
3. Desarrollo paso a paso, numerado.
4. Conclusión final que repita el resultado y nombre el criterio usado.
No uses emojis ni etiquetas decorativas dentro del desarrollo.

FORMATO VISUAL PARA TELEGRAM Y RELOJ (SIN LATEX CRUDO)
- PROHIBIDO usar LaTeX crudo (prohibido \\frac, \\sum, \\lim, \\int, \\sqrt, \\left, \\right, $$) y PROHIBIDO envolver fórmulas en comillas invertidas (`) o bloques de código.
- Escribe TODA expresión matemática en Unicode limpio, con líneas claras y verticales (una igualdad por línea cuando sea una cadena):
  ∑ₙ₌₁^∞ aₙ, lím_{n→∞}, ∫, √(...), ³√(...), x², xⁿ, aₙ, aₙ₊₁, sₙ, π, ∞, ≤, ≥, ≠, ⇒, →, ·, |...|, sen(x), cos(x), tan(x), ln(n), arctan(n).

REGLAS GENERALES (OBLIGATORIAS)
- Nunca saltes un paso algebraico. Si simplificas, muestra la línea intermedia.
- Nunca pierdas lím_{n→∞} ni las barras de valor absoluto |...| a mitad de una cadena de igualdades: escríbelos en TODAS y cada una de las líneas "=" hasta el paso exacto en que evalúas el límite.
- Antes de usar un criterio, verifica TODAS sus hipótesis y escríbelas (por ejemplo: términos positivos, continuidad, decreciente, límite finito y distinto de 0).
- Cada "⇒" debe tener una razón visible. Si usas una propiedad (álgebra de límites, monotonía de la raíz, etc.), nómbrala.
- Usa solo teoremas del curso: definición de límite, álgebra de límites, Teorema del Sándwich, Teorema de Weierstrass (convergencia monótona), inducción, criterios de monotonía (diferencia, cociente, derivada), serie geométrica, telescópica, serie p, serie armónica, criterio de divergencia, criterio de la integral, comparación directa, comparación por paso al límite, criterio de Leibniz, convergencia absoluta y condicional, criterio de la razón, series de potencias. Si un resultado necesita un teorema fuera de esta lista (por ejemplo, el Teorema de Abel), no lo uses en la demostración: dilo en una nota aparte.
- Nunca atribuyas algo al profesor o a "la cátedra" si no te lo dieron explícitamente.
- No operes con ∞ como si fuera un número (no escribas "1/∞ = 0"). Escribe "lím 1/n = 0".
- Distingue "dividir numerador y denominador por n" (no cambia la fracción) de "dividir los miembros de una desigualdad por n > 0" (no cambia el sentido). No confundas las dos cosas.
- En una cadena de igualdades, no pongas justificaciones entre dos signos "=". Ponlas al final, entre paréntesis.

SUCESIONES
- Forma ∞/∞: divide numerador y denominador por la mayor potencia de n, y escribe la división explícitamente.
- Forma ∞ − ∞: multiplica por el conjugado, usa (A − B)(A + B) = A² − B² y, al meter n dentro de la raíz, justifica que √(n²) = n porque n > 0.
- Sándwich: escribe las dos cotas, justifica la desigualdad y calcula el límite de cada cota.
- Inducción: escribe caso base, hipótesis, tesis y paso inductivo. En el paso inductivo, escribe explícitamente la igualdad de la recurrencia que usas (por ejemplo, √(6 + aₖ) = aₖ₊₁).
- Criterio del cociente para monotonía: primero demuestra que aₙ > 0. Justifica por qué el cociente es menor (o mayor) que 1.
- Weierstrass: cita la monotonía y la cota ya probadas; al despejar el límite L, justifica qué solución se descarta y por qué.

SERIES NUMÉRICAS
- Geométrica: muestra que aₙ₊₁/aₙ es constante (no depende de n), identifica el primer término y r, verifica |r| < 1 (o |r| ≥ 1 si diverge) y aplica S = a/(1 − r) cuando corresponda.
- Telescópica: haz fracciones parciales mostrando cómo se obtienen A y B. Escribe varios términos de sₙ, indica qué se cancela y cuántos términos sobreviven al inicio y al final. Calcula lím sₙ.
- Criterio de divergencia: calcula lím aₙ y concluye solo si es distinto de 0. Si da 0, el criterio no concluye nada.
- Comparación por paso al límite: explica de dónde sale bₙ (términos dominantes), verifica que ambas series son de términos positivos, calcula L paso a paso, escribe "0 < L < ∞" y justifica por qué ∑ bₙ converge o diverge (serie p con su valor de p, armónica o geométrica).
- Criterio de la integral: define f con f(n) = aₙ y verifica que es continua, positiva y decreciente (con derivada). Escribe la integral impropia como lím_{b→∞}, calcula la primitiva completa antes de evaluar y justifica el límite. Aclara que el valor de la integral NO es la suma de la serie.
- Leibniz: verifica las tres condiciones: bₙ > 0, bₙ → 0 y bₙ decreciente (justificando la desigualdad). Si bₙ decrece solo desde cierto n, dilo.
- Convergencia absoluta o condicional: estudia ∑|aₙ| por separado y concluye cuál de las dos es.

SERIES DE POTENCIAS
- Identifica el centro a desde la forma (x − a)ⁿ.
- Aplica el criterio de la razón a la serie COMPLETA, con x incluida y con valor absoluto. Escribe aₙ₊₁ y aₙ por separado antes de simplificar y agrupa factor por factor. No uses la fórmula R = 1/L de los coeficientes.
- Despeja el intervalo mostrando cada paso (por ejemplo: |x − 1| < 2 ⇒ −2 < x − 1 < 2 ⇒ −1 < x < 3).
- Revisa CADA extremo por separado: reemplaza x, simplifica los signos con cuidado ((−1)ⁿ·(−1)ⁿ = 1) e indica el criterio que usas.
- Después de cada extremo, escribe si se incluye o no. Arma el intervalo final con el corchete o paréntesis que corresponde a cada extremo, y verifica que coincide con lo concluido.
- Si el límite es 0 para todo x, concluye R = ∞; no confundas L = 0 con R = 0.

TAYLOR / MACLAURIN
- Escribe la definición, calcula las derivadas y su valor en a, identifica el patrón con una fórmula y arma la serie con el índice correcto (por ejemplo, x²ⁿ y (2n)!).
- En el criterio de la razón, muestra la simplificación de factoriales: (2n+2)! = (2n+2)(2n+1)·(2n)!.

INTEGRALES, ÁREAS Y VOLÚMENES (SI EL EJERCICIO ES DE INTEGRACIÓN)
- Aplica el mismo formato 1-2-3-4 sin emojis y sin saltarte ningún paso algebraico.
- Escribe explícitamente u, du, cambio de límites en integrales definidas, fracciones parciales paso a paso, y regla de Barrow completa.
- Si es integral impropia (incluidas discontinuidades en extremos o puntos interiores), escríbela formalmente con lím_{t→...} en toda la cadena hasta evaluar.
- En áreas y volúmenes por secciones transversales, verifica si las curvas se cruzan en el intervalo y parte la integral en los cruces.

ANTES DE RESPONDER, VERIFICA
- ¿Nombré el criterio y verifiqué todas sus hipótesis?
- ¿Hay algún salto algebraico sin línea intermedia?
- ¿El resultado final coincide con lo que concluí en el desarrollo (sobre todo el intervalo)?
- ¿Usé algún teorema que no está en la lista del curso?
- ¿Atribuí algo al profesor sin que me lo dijeran?
"""

ENFORCEMENT_SUFFIX = (
    "\n\n[VERIFICACIÓN OBLIGATORIA ANTES DE EMITIR LA RESPUESTA: "
    "1) Usa el formato exacto: 1. Resultado, 2. Criterio, 3. Desarrollo paso a paso numerado (1., 2., 3., ...), 4. Conclusión final, SIN emojis ni LaTeX crudo ni comillas invertidas. "
    "2) Verifica y escribe TODAS las hipótesis del criterio antes de aplicarlo. "
    "3) NO saltes NINGÚN paso algebraico ni línea intermedia: en TODO límite ∞/∞ (incluso dentro del criterio de la razón, como lím n/(n+1)) divide explícitamente numerador y denominador por la mayor potencia de n; en el criterio de la razón agrupa factor por factor antes de simplificar y mantén lím y |...| en TODA la cadena de igualdades hasta evaluar; justifica cada '⇒' y cada desigualdad (ej. 1/(n+1) < 1/n porque n+1 > n > 0). "
    "4) En una cadena de igualdades no pongas justificaciones entre dos signos '=' (ponlas al final de la línea entre paréntesis). Nunca operes con ∞ como número (escribe lím 1/n = 0) y nunca atribuyas nada al profesor ni a la cátedra si no se mencionó en la pregunta. "
    "5) En series de potencias aplica criterio de la razón a la serie COMPLETA con x y |...| (NO R=1/L), despeja paso a paso justificando cada '⇒' y analiza ambos extremos por separado confirmando que el intervalo final coincide.]"
)

AUDITOR_SYSTEM_PROMPT = f"""Eres un corrector universitario extremadamente estricto de Cálculo Integral (Pedagogía en Educación Matemática).
Recibirás una PREGUNTA y un BORRADOR de solución. Tu ÚNICA tarea es auditar el borrador contra TODAS las reglas de abajo y reescribir la solución COMPLETA corrigiendo absolutamente TODOS los saltos algebraicos, hipótesis faltantes o errores de formato, entregando DIRECTAMENTE la solución final corregida (sin prefacios, sin decir "borrador corregido", sin comentarios meta).

CHECKLIST CRÍTICO QUE DEBES EXPANDIR SI EL BORRADOR LO RESUMIÓ:
- Formato obligatorio exacto: 1. Resultado / 2. Criterio / 3. Desarrollo paso a paso (CON PASOS NUMERADOS 1., 2., 3., ...) / 4. Conclusión final. CERO emojis, CERO LaTeX crudo, CERO comillas invertidas (`).
- Si el borrador calculó aₙ₊₁/aₙ o aₙ/bₙ aparte SIN lím o SIN |...| antes de aplicar el límite, intégralo en una cadena formal con lím (y |...| cuando corresponda) en TODAS las igualdades hasta evaluar.
- Si en el criterio de la razón (numérica o de potencias) el borrador no agrupó factor por factor antes de simplificar (ej. [(-1)ⁿ⁺¹/(-1)ⁿ]·[(x−1)ⁿ⁺¹/(x−1)ⁿ]·[n/(n+1)]·[2ⁿ/2ⁿ⁺¹]), agrega explícitamente esa línea y la simplificación de cada factor.
- Si el borrador evaluó CUALQUIER límite ∞/∞ (incluyendo lím n/(n+1) dentro de una serie de potencias) sin escribir explícitamente la división del numerador y denominador por la mayor potencia de n y sin escribir lím 1/n = 0, agrega todas esas líneas intermedias.
- Si el borrador puso un "⇒" sin razón visible, agrega su justificación entre paréntesis (ej: multiplicando ambos miembros por 2 > 0; definición de valor absoluto; sumando 1 a los tres miembros; álgebra de límites).
- Si el borrador usó una desigualdad sin justificar (como 1/(n+1) < 1/n en Leibniz, o √(n²) = n porque n > 0), agrega la justificación explícita.
- Si el borrador puso explicaciones entre dos signos "=", muévelas al final de la línea entre paréntesis.
- Verifica que NO aparezca 1/∞ = 0, que NO haya atribuciones al profesor o a la cátedra si la pregunta no las mencionó, que TODOS los teoremas pertenezcan a la lista permitida del curso, y que el intervalo final (corchetes/paréntesis) coincida 100% con el análisis de los extremos.

REGLAS COMPLETAS A CUMPLIR AL 100%:
{BASE_SYSTEM_INSTRUCTION}"""

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
    texto = re.sub(r"(?<!\S)\*(?!\s)(.+?)(?<!\s)\*(?!\S)", r"<i>\1</i>", texto)

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

def dividir_en_chunks_markdown(texto: str, max_len: int = 3400) -> list[str]:
    if len(texto) <= max_len:
        return [texto]
    parrafos = texto.split("\n\n")
    unidades = []
    for p in parrafos:
        if len(p) <= max_len:
            unidades.append(p)
        else:
            sub = []
            sub_len = 0
            for linea in p.split("\n"):
                if sub_len + len(linea) + 1 > max_len and sub:
                    unidades.append("\n".join(sub))
                    sub, sub_len = [linea], len(linea)
                else:
                    sub.append(linea)
                    sub_len += len(linea) + 1
            if sub:
                unidades.append("\n".join(sub))

    chunks, chunk_actual, len_actual = [], [], 0
    for p in unidades:
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
    relevant_docs = SEARCH_ENGINE.search(message_text, top_k=4)
    system_prompt = (
        "=== MATERIAL DE REFERENCIA INDEXADO ===\n"
        "(Úsalo SOLO para leer enunciados si la estudiante pregunta por un ejercicio del certamen, guía, listado o integradora. "
        "PROHIBIDO copiar su formato resumido, PROHIBIDO saltarte pasos porque estén abreviados aquí, y PROHIBIDO atribuir algo "
        "al profesor o a 'la cátedra' si no se mencionó explícitamente en la consulta.)\n"
        f"{relevant_docs}\n"
        "=== FIN DEL MATERIAL DE REFERENCIA ===\n\n"
        f"{BASE_SYSTEM_INSTRUCTION}"
    )

    if user_id not in user_histories:
        user_histories[user_id] = []
    if len(user_histories[user_id]) > 4:
        user_histories[user_id] = user_histories[user_id][-4:]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(user_histories[user_id])
    user_payload_text = f"{message_text}{ENFORCEMENT_SUFFIX}" if use_reasoner else message_text
    messages.append({"role": "user", "content": user_payload_text})

    model_name = "deepseek-reasoner" if use_reasoner else "deepseek-chat"

    if DEEPSEEK_API_KEY:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 16384 if use_reasoner else 4096,
        }
        if not use_reasoner:
            payload["temperature"] = 0.1

        resp = requests.post(DEEPSEEK_BASE_URL, headers=headers, json=payload, timeout=180)
        if resp.status_code != 200:
            raise RuntimeError(f"DeepSeek API ({resp.status_code}): {resp.text[:300]}")
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]

        # Segunda pasada de auditoría estricta para ejercicios: expande cualquier salto algebraico,
        # hipótesis omitida, límite sin dividir por la mayor potencia o '⇒' sin razón.
        if use_reasoner and reply.strip():
            try:
                audit_payload = {
                    "model": "deepseek-reasoner",
                    "messages": [
                        {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"PREGUNTA DE LA ESTUDIANTE:\n{message_text}\n\n"
                                f"BORRADOR A AUDITAR Y EXPANDIR SIN SALTAR NINGÚN PASO:\n{reply}"
                            ),
                        },
                    ],
                    "max_tokens": 16384,
                }
                audit_resp = requests.post(DEEPSEEK_BASE_URL, headers=headers, json=audit_payload, timeout=180)
                if audit_resp.status_code == 200:
                    audited = audit_resp.json()["choices"][0]["message"]["content"]
                    if audited and audited.strip():
                        reply = audited
            except Exception as audit_err:
                logger.warning(f"Auditoría segunda pasada omitida por error: {audit_err}")

        user_histories[user_id].append({"role": "user", "content": message_text})
        user_histories[user_id].append({"role": "assistant", "content": reply})
        return reply

    if GEMINI_API_KEY and genai is not None:
        model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_prompt)
        res = model.generate_content(user_payload_text)
        return res.text

    raise RuntimeError("Configura tu DEEPSEEK_API_KEY en el archivo .env o en Render.")

async def ask_with_typing(update: Update, user_id: int, prompt: str, use_reasoner: bool = False) -> str:
    stop_event = asyncio.Event()

    async def keep_typing():
        while not stop_event.is_set():
            try:
                await update.effective_chat.send_action(ChatAction.TYPING)
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                pass

    typing_task = asyncio.create_task(keep_typing())
    try:
        return await asyncio.to_thread(ask_deepseek, user_id, prompt, use_reasoner)
    finally:
        stop_event.set()
        await typing_task

async def split_and_send(update: Update, text: str):
    for chunk in dividir_en_chunks_markdown(text, max_len=3400):
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
            f"Resuelve completo y sin saltarte pasos el Problema '{num}' del Certamen 3 – Series Infinitas (2025-2)."
        )
    else:
        prompt = (
            "Haz un resumen compacto para mi Galaxy Watch 4 con los 6 problemas del Certamen 3 – Series Infinitas (2025-2, Prof. Eligio Colmenares): "
            "1. 3 series numéricas (geométrica, divergencia, álgebra de series), 2. Telescópica 3/(n(n+3)) por {sₙ}, "
            "3. Comparación en el límite con 1/√n, 4. 3 series de potencias (centro, radio e intervalo), "
            "5. Demostración de 1/(1-x) y tan⁻¹(x), 6. Serie de Taylor de cos(x) en a=0 + R e I, "
            "y dime que puedo escribir /c3 [1a..6] para ver cualquier solución paso a paso."
        )
    try:
        reply = await ask_with_typing(update, user_id, prompt, bool(num))
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def integradora_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    num = " ".join(context.args) if context.args else ""
    if num:
        prompt = (
            f"Resuelve completo y sin saltarte pasos el Problema '{num}' de la Evaluación Integradora de Cálculo Integral."
        )
    else:
        prompt = (
            "Haz un resumen compacto para mi Galaxy Watch 4 con los 5 problemas de la Evaluación Integradora de Cálculo Integral (Prof. Eligio Colmenares): "
            "1. Las 6 integrales (1a–1f, incluyendo impropias y fracciones parciales), 2. Área con cruce y Volumen por secciones transversales cuadradas, "
            "3. Sucesión recurrente aₙ₊₁=½(aₙ+2/aₙ), 4. Series por Comparación límite, Razón y Raíz, y 5. Radio e intervalo de convergencia, "
            "y dime que puedo escribir /integradora [1a..5] o /practicar integradora."
        )
    try:
        reply = await ask_with_typing(update, user_id, prompt, bool(num))
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    num = " ".join(context.args) if context.args else ""
    if num:
        prompt = (
            f"Resuelve completo y sin saltarte pasos el ejercicio '{num}' de la Guía de preparación para el test de la Unidad IV (Sucesiones y Series Infinitas)."
        )
    else:
        prompt = (
            "Haz un resumen ultra compacto para mi Galaxy Watch 4 de las secciones A hasta L (ejercicios 1 al 48) "
            "de la Guía de preparación para el test de la Unidad IV (Series Infinitas) del Prof. Eligio Colmenares, "
            "y dime que puedo escribir /test [número] (ej: /test 1, /test 15, /test 48) para ver cualquier ejercicio resuelto."
        )
    try:
        reply = await ask_with_typing(update, user_id, prompt, bool(num))
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def listado_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    num = " ".join(context.args) if context.args else ""
    if num:
        prompt = (
            f"Resuelve completo y sin saltarte pasos el ejercicio '{num}' del Listado de Cálculo Integral de la Unidad IV (Sucesiones y Series Infinitas 2026-2)."
        )
    else:
        prompt = (
            "Haz un resumen compacto para mi Galaxy Watch 4 de los 22 ejercicios del Listado de la Unidad IV "
            "(Sucesiones: ejercicios 1 al 13; Series Infinitas y de Potencias: ejercicios 12 al 22) del Prof. Eligio Colmenares, "
            "y dime que puedo escribir /listado [número] (ej: /listado 5, /listado 14, /listado 22) para resolver cualquiera."
        )
    try:
        reply = await ask_with_typing(update, user_id, prompt, bool(num))
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
    try:
        reply = await ask_with_typing(update, user_id, q, True)
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    try:
        reply = await ask_with_typing(update, user_id, user_text, True)
        await split_and_send(update, reply)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    caption = update.message.caption or "Resuelve este ejercicio completo, justificado y sin saltarte ningún paso."
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
        reply = await ask_with_typing(update, user_id, full_query, True)
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
