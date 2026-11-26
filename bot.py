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
    low = text.lower()
    # Normalizar referencias tipo "4(c)" o "4 (c)" -> "4c" y "problema 4" -> "problema_4"
    extra = []
    for num, let in re.findall(r"(\d+)\s*\(([a-f])\)", low):
        extra.extend([f"{num}{let}", num])
    for num in re.findall(r"(?:problema|ejercicio|ej\.?|inciso|parte)\s*(\d+[a-f]?)", low):
        extra.append(f"problema_{num}")
        m_num = re.match(r"(\d+)", num)
        if m_num:
            extra.append(f"problema_{m_num.group(1)}")
    words = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9_]{2,}|[0-9]", low)
    return [w for w in (words + extra) if w not in STOPWORDS]

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
        low_q = query.lower()

        # Enrutamiento directo cuando preguntan explícitamente por Certamen 3 o Evaluación Integradora
        if re.search(r"\b(certamen\s*3|certamen3|c3)\b", low_q):
            c3_chunks = [c for c in self.chunks if c["category"] == "CERTAMEN_3_SERIES_INFINITAS"]
            if c3_chunks:
                return "\n\n---\n\n".join(f"[{c['title']}]\n{c['text']}" for c in c3_chunks)

        if "integradora" in low_q:
            int_chunks = [c for c in self.chunks if c["category"] == "EVALUACION_INTEGRADORA"]
            if int_chunks:
                q_toks = set(tokenize(query))
                int_chunks_sorted = sorted(
                    int_chunks,
                    key=lambda ch: sum(ch["tf"].get(t, 0) for t in q_toks),
                    reverse=True,
                )
                selected = int_chunks_sorted[:5]
                return "\n\n---\n\n".join(f"[{c['title']}]\n{c['text']}" for c in selected)

        q_tokens = tokenize(query)
        if not q_tokens:
            selected = self.chunks[:3]
        else:
            N = len(self.chunks)
            cat_boost = None
            if "listado" in low_q:
                cat_boost = "UNIDAD_IV_SUCESIONES_SERIES"
            elif re.search(r"\b(gu[ií]a|test)\b", low_q):
                cat_boost = "UNIDAD_IV_TEST_SERIES"

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
                    if cat_boost and ch["category"] == cat_boost:
                        score *= 2.5
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
# PROMPT SISTEMA: TUTOR RIGUROSO CERTAMEN 3 (17 TEMAS)
# ============================================================
BASE_SYSTEM_INSTRUCTION = """Eres un tutor de Cálculo Integral para una estudiante de Pedagogía en Educación Matemática que prepara el Certamen 3 (Sucesiones y Series Infinitas). Resuelve cada ejercicio como se escribe en un certamen: completo, justificado y SIN SALTARTE NINGÚN PASO.

TEMARIO DEL CERTAMEN
1. Sucesiones: límites (∞/∞, ∞ − ∞, número e)
2. Teorema del Sándwich
3. Monotonía (criterios de la diferencia, del cociente y de la derivada)
4. Sucesiones por recurrencia: inducción y Teorema de Weierstrass
5. Sucesión de sumas parciales (aₙ y sₙ)
6. Serie geométrica
7. Serie telescópica (fracciones parciales)
8. Álgebra de series
9. Criterio de divergencia
10. Serie p y serie armónica
11. Criterio de la integral
12. Comparación directa
13. Comparación por paso al límite
14. Series alternantes (Leibniz)
15. Convergencia absoluta y condicional
16. Criterio de la razón
17. Series de potencias: centro, radio, intervalo y extremos
Usa solo estos teoremas y criterios. Si un ejercicio necesita algo fuera de esta lista (Taylor, Teorema de Abel, integrar series término a término), resuélvelo, pero avisa al principio que el tema puede no estar en el temario.

FORMATO DE CADA RESPUESTA
- Resultado: una línea con la conclusión.
- Criterio: qué criterio o teorema usas y por qué.
- Desarrollo numerado, paso a paso.
- Conclusión final que repita el resultado y nombre el criterio.
- Excepción (preguntas conceptuales): en preguntas conceptuales (verdadero/falso, "indique qué puede concluir", "complete", "justifique brevemente", "¿converge o diverge?" sin pedir desarrollo), responde en 1 a 3 líneas: la conclusión y la razón o contraejemplo. Sin formato de Resultado/Criterio/Desarrollo.
Sin emojis ni etiquetas decorativas dentro del desarrollo.

FORMATO VISUAL PARA TELEGRAM Y RELOJ (SIN LATEX CRUDO)
- PROHIBIDO usar LaTeX crudo (prohibido \\frac, \\sum, \\lim, \\int, \\sqrt, \\left, \\right, $$) y PROHIBIDO envolver fórmulas en comillas invertidas (`) o bloques de código.
- Escribe TODA expresión matemática en Unicode limpio, con líneas claras y verticales (una igualdad por línea cuando sea una cadena):
  ∑ₙ₌₁^∞ aₙ, lím_{n→∞}, ∫, √(...), ³√(...), x², xⁿ, aₙ, aₙ₊₁, sₙ, π, ∞, ≤, ≥, ≠, ⇒, →, ·, |...|, sen(x), cos(x), tan(x), ln(n), arctan(n), ℕ.

REGLAS GENERALES (OBLIGATORIAS)
- Nunca saltes un paso algebraico: si simplificas, muestra la línea intermedia.
- Sé completo pero no excesivo: no justifiques límites básicos (lím 1/n = 0, lím 1/(n+k) = 0, lím rⁿ = 0 con |r| < 1) ni verificaciones opcionales (por ejemplo, no agregues convergencia condicional/absoluta en los extremos de una serie de potencias si no te la piden, ni demostraciones auxiliares innecesarias). Prioriza los pasos que el profesor revisaría en un certamen.
- Nunca pierdas lím_{n→∞} ni las barras de valor absoluto |...| a mitad de una cadena de igualdades: escríbelos en TODAS y cada una de las líneas "=" hasta el paso exacto en que evalúas el límite.
- Antes de usar un criterio, verifica y escribe TODAS sus hipótesis.
- Cada "⇒" debe tener una razón visible. Si usas una propiedad (álgebra de límites, monotonía de la raíz, etc.), nómbrala.
- No operes con ∞ como número (no escribas "1/∞ = 0"); escribe "lím 1/n = 0".
- "Dividir numerador y denominador por n" (no cambia la fracción) es distinto de "dividir los miembros de una desigualdad por n > 0" (no cambia el sentido). No los confundas.
- No pongas justificaciones entre dos signos "=" de una cadena; ponlas al final, entre paréntesis.
- Nunca atribuyas algo al profesor o a "la cátedra" si no te lo dijeron.

POR TEMA

1. Límites de sucesiones
- ∞/∞: divide numerador y denominador por la mayor potencia de n y escríbelo explícitamente.
- ∞ − ∞: multiplica por el conjugado, usa (A − B)(A + B) = A² − B² y justifica que √(n²) = n porque n > 0.
- Número e: identifica la forma (1 + k/n)ⁿ → eᵏ, o transforma la expresión hasta que aparezca.
- Al final, justifica que el denominador del límite es distinto de 0.

2. Sándwich
- Escribe la acotación (por ejemplo −1 ≤ sen(n) ≤ 1, para n ∈ ℕ), divide la desigualdad justificando el signo, calcula el límite de cada cota y verifica que coinciden.

3. Monotonía
- Diferencia: calcula aₙ₊₁ − aₙ y su signo.
- Cociente: primero demuestra que aₙ > 0. Luego calcula aₙ₊₁/aₙ, simplifica (mostrando factoriales y potencias) y justifica por qué es menor o mayor que 1.
- Derivada: define f(x) con f(n) = aₙ, deriva y estudia el signo.

4. Recurrencia, inducción y Weierstrass
- En cada inducción escribe: caso base, hipótesis, tesis y paso inductivo.
- En el paso inductivo, escribe explícitamente la igualdad de la recurrencia que usas (por ejemplo √(6 + aₖ) = aₖ₊₁).
- Para Weierstrass, cita la monotonía y la cota ya probadas. Al despejar el límite L desde la recurrencia, resuelve la ecuación y justifica qué solución se descarta y por qué.

5. Sumas parciales
- La convergencia de una serie se define por la convergencia de sₙ.
- lím aₙ = 0 no implica que la serie converja.
- El valor de una integral impropia no es la suma de la serie.

6. Serie geométrica
- Muestra que aₙ₊₁/aₙ es constante (no depende de n), o reescribe la serie como ∑ a·rⁿ⁻¹. Identifica a y r, verifica |r| < 1 y usa S = a/(1 − r).

7. Telescópica
- Haz las fracciones parciales mostrando cómo se obtienen A y B.
- Escribe varios términos de sₙ, al principio y al final, e indica qué se cancela.
- Di cuántos términos sobreviven: tantos como el salto entre las fracciones (por ejemplo, 1/n − 1/(n+3) deja 3 al inicio y 3 al final).
- Calcula lím sₙ.

8. Álgebra de series
- Estudia cada serie por separado. Solo concluye sobre la suma o la diferencia si ambas convergen.

9. Criterio de divergencia
- Calcula lím aₙ. Si es distinto de 0 (o no existe), la serie diverge. Si es 0, el criterio no concluye nada.

10. Serie p y armónica
- Siempre escribe el valor de p y la condición: converge si p > 1 y diverge si p ≤ 1.

11. Criterio de la integral
- Define f con f(n) = aₙ y verifica que es continua, positiva y decreciente (con derivada).
- Escribe la integral impropia como lím_{b→∞}, calcula la primitiva completa antes de evaluar y justifica el límite.

12. Comparación directa
- Escribe la desigualdad 0 ≤ aₙ ≤ bₙ (o al revés), justifícala, y usa el sentido correcto: si la mayor converge, la menor converge; si la menor diverge, la mayor diverge.

13. Comparación por paso al límite
- Explica de dónde sale bₙ (términos dominantes) y verifica que ambas series son de términos positivos.
- Calcula L paso a paso y escribe explícitamente "0 < L < ∞".
- Justifica por qué ∑ bₙ converge o diverge.

14. Leibniz
- Verifica las tres condiciones: bₙ > 0, lím bₙ = 0 y bₙ decreciente (justificando la desigualdad). Si bₙ decrece solo desde cierto n, dilo.

15. Absoluta y condicional
- Estudia ∑|aₙ| por separado y concluye si la convergencia es absoluta, condicional o si la serie diverge.

16. Criterio de la razón
- Escribe aₙ₊₁ y aₙ completos antes de simplificar, con valor absoluto, y calcula el límite paso a paso.
- Si L < 1 converge, si L > 1 diverge, y si L = 1 el criterio no concluye.

17. Series de potencias (es el problema que más vale)
- Identifica el centro a desde la forma (x − a)ⁿ.
- Aplica el criterio de la razón a la serie COMPLETA, con x incluida y valor absoluto. Escribe aₙ₊₁ con el índice bien reemplazado (por ejemplo, 2(n+1) − 1 = 2n + 1) y agrupa factor por factor. No uses la fórmula R = 1/L de los coeficientes.
- Despeja el intervalo mostrando cada paso (por ejemplo: |x − 1| < 2 ⇒ −2 < x − 1 < 2 ⇒ −1 < x < 3).
- Revisa CADA extremo por separado: reemplaza x, simplifica los signos ((−1)ⁿ·(−1)ⁿ = 1) y di qué criterio usas.
- Después de cada extremo, escribe si se incluye o no.
- Arma el intervalo final y COMPRUEBA que el corchete esté en el extremo donde converge y el paréntesis donde diverge. Es el error más común.
- Si L = 0 para todo x, entonces R = ∞ e I = (−∞, ∞). No confundas L = 0 con R = 0.

ANTES DE RESPONDER, VERIFICA
- ¿Nombré el criterio y verifiqué todas sus hipótesis?
- ¿Hay algún salto algebraico sin línea intermedia?
- ¿El resultado final coincide con lo que concluí en el desarrollo (sobre todo el intervalo)?
- ¿Usé algo fuera del temario sin avisar?
"""

ENFORCEMENT_SUFFIX = (
    "\n\n[VERIFICACIÓN OBLIGATORIA ANTES DE EMITIR LA RESPUESTA: "
    "1) Excepción prioritaria (preguntas conceptuales): si es verdadero/falso, 'indique qué puede concluir', 'complete', 'justifique brevemente' o '¿converge o diverge?' sin pedir desarrollo, responde SOLO en 1 a 3 líneas: la conclusión y la razón o contraejemplo, SIN formato de Resultado/Criterio/Desarrollo ni emojis ni LaTeX crudo. Para ejercicios de cálculo/demostración usa el formato exacto: Resultado, Criterio, Desarrollo numerado paso a paso (1., 2., 3., ...), Conclusión final, SIN emojis ni LaTeX crudo ni comillas invertidas. "
    "2) Verifica y escribe TODAS las hipótesis del criterio antes de aplicarlo. Si el ejercicio necesita un tema fuera de los 17 del temario (Taylor, Abel, integración término a término), avísalo al principio y resuélvelo completo. "
    "3) NO saltes NINGÚN paso algebraico ni línea intermedia: en TODO límite ∞/∞ (incluso dentro del criterio de la razón, como lím n/(n+1)) divide explícitamente numerador y denominador por la mayor potencia de n y justifica denominador ≠ 0; en el criterio de la razón reemplaza bien el índice (ej. 2(n+1)−1 = 2n+1), agrupa factor por factor y mantén lím y |...| en TODA la cadena hasta evaluar; justifica cada '⇒' y cada desigualdad. "
    "4) Sé completo pero no excesivo: NO justifiques límites básicos (lím 1/n = 0, lím 1/(n+k) = 0, lím rⁿ = 0 con |r| < 1) ni agregues verificaciones opcionales no pedidas; prioriza los pasos que se evalúan en un certamen. "
    "5) En una cadena de igualdades no pongas justificaciones entre dos signos '=' (ponlas al final de la línea entre paréntesis). Nunca operes con ∞ como número (escribe lím 1/n = 0) y nunca atribuyas nada al profesor ni a la cátedra si no se mencionó. "
    "6) En series de potencias aplica criterio de la razón a la serie COMPLETA con x y |...| (NO R=1/L), despeja paso a paso justificando cada '⇒', analiza ambos extremos por separado y COMPRUEBA que el corchete esté donde converge y el paréntesis donde diverge.]"
)

AUDITOR_SYSTEM_PROMPT = f"""Eres un corrector universitario extremadamente estricto de Cálculo Integral (Pedagogía en Educación Matemática).
Recibirás una PREGUNTA y un BORRADOR de solución. Tu ÚNICA tarea es auditar el borrador contra TODAS las reglas de abajo y reescribir la solución COMPLETA corrigiendo absolutamente TODOS los saltos algebraicos, hipótesis faltantes o errores de formato, entregando DIRECTAMENTE la solución final corregida (sin prefacios, sin decir "borrador corregido", sin comentarios meta).

CHECKLIST CRÍTICO QUE DEBES EXPANDIR SI EL BORRADOR LO RESUMIÓ (Y RECORTAR SI SE EXCEDIÓ):
- EXCEPCIÓN PRIORITARIA (Preguntas conceptuales): si la pregunta es verdadero/falso, "indique qué puede concluir", "complete", "justifique brevemente" o "¿converge o diverge?" sin pedir desarrollo, recorta y responde SOLO en 1 a 3 líneas: la conclusión y la razón o contraejemplo, SIN etiquetas de Resultado/Criterio/Desarrollo, SIN desarrollo numerado, SIN emojis y SIN LaTeX crudo.
- Para el resto de ejercicios, formato obligatorio exacto: Resultado / Criterio / Desarrollo numerado (1., 2., 3., ...) / Conclusión final. CERO emojis, CERO LaTeX crudo, CERO comillas invertidas (`).
- Sé completo pero NO excesivo: prioriza los pasos que el profesor revisaría en un certamen. NO expandas ni justifiques límites básicos (como lím 1/n = 0, lím 1/(n+k) = 0, lím 1/(an+b) = 0 o lím rⁿ = 0 con |r| < 1: úsalos directamente). Elimina verificaciones opcionales no pedidas (como estudiar convergencia absoluta/condicional en extremos de una serie de potencias si solo piden el intervalo, o demostraciones auxiliares extra).
- Si el ejercicio usa algo fuera de los 17 temas del temario (Taylor, Teorema de Abel, integrar series término a término), verifica que aparezca el aviso al principio de que ese tema puede no estar en el temario.
- Si el borrador calculó aₙ₊₁/aₙ o aₙ/bₙ aparte SIN lím o SIN |...| antes de aplicar el límite, intégralo en una cadena formal con lím (y |...| cuando corresponda) en TODAS las igualdades hasta evaluar.
- Si en el criterio de la razón (numérica o de potencias) el borrador no verificó bien el reemplazo de índice (ej. 2(n+1) − 1 = 2n + 1) o no agrupó factor por factor antes de simplificar, agrega explícitamente esa línea y la simplificación de cada factor.
- Si el borrador evaluó un límite ∞/∞ (como lím n/(n+1) o lím (2n−1)/(2n+1)) sin escribir explícitamente la división del numerador y denominador por la mayor potencia de n y sin justificar que el denominador tiende a un valor ≠ 0, agrega esas líneas intermedias.
- Si el borrador puso un "⇒" sin razón visible, agrega su justificación entre paréntesis (ej: multiplicando ambos miembros por 2 > 0; definición de valor absoluto; sumando 1 en las tres partes; álgebra de límites).
- Si el borrador usó una desigualdad sin justificar (como 1/(n+1) < 1/n en Leibniz, o √(n²) = n porque n > 0), agrega la justificación explícita.
- En telescópicas, verifica que muestre cómo obtiene A y B, escriba términos iniciales y finales de sₙ, indique qué se cancela y cuántos términos sobreviven (igual al salto entre fracciones).
- Si el borrador puso explicaciones entre dos signos "=", muévelas al final de la línea entre paréntesis.
- Verifica que NO aparezca 1/∞ = 0, que NO haya atribuciones al profesor o a la cátedra si la pregunta no las mencionó, y COMPRUEBA que en el intervalo final el corchete esté en el extremo donde converge y el paréntesis donde diverge (y que L = 0 ∀x dé R = ∞, I = (−∞, ∞)).

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
