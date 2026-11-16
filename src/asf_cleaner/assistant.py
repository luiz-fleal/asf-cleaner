"""The map's AI bar: answers questions about the project and drives the page.

Runs on Amazon Bedrock with a small, cheap model (Amazon Nova Lite by default) through the Converse API, authenticated
with a bearer API key (BEDROCK_API_KEY, from the environment or the `.env` file). The model answers from a knowledge
base written into the system prompt plus the state of the page the client sends, and may call one tool, `control_map`,
to change layers, open a region, run a forecast and so on. Every action it asks for is checked here against a
whitelist and clamped, so the page only ever receives actions it knows how to run.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from urllib.parse import quote

import requests

import re

from .assistant_places import PLACES
from .config import env_value

DEFAULT_MODEL = "amazon.nova-lite-v1:0"
MAX_QUESTION = 600
MAX_CONTEXT = 2500
MAX_HISTORY = 6
MAX_ACTIONS = 6
LANGS = {"en": "English", "pt": "Brazilian Portuguese", "es": "Spanish"}

LAYERS = {"nisar", "water", "dist", "imerg", "lst", "smap"}
REGIONS = {"amazon", "sao-paulo", "rio-grande-do-sul"}
BASES = {"satellite", "bluemarble", "modis"}
TABS = {"data", "forecast", "regions"}
KINDS = {"earthquake", "flood", "wildfire", "storm"}
# name -> (min, max) for the numbers a simulation may carry
PARAMS = {"magnitude": (5, 9), "depth": (1, 100), "rise": (1, 15), "hours": (1, 72), "wind": (0, 30), "windDir": (0, 360),
          "dryness": (0.4, 1.6), "gust": (60, 300), "track": (0, 360)}
# the region the map covers: South and Central America
LAT_RANGE, LON_RANGE = (-58.0, 24.0), (-120.0, -30.0)

_KNOWLEDGE = """\
SARIS is an interactive 3D observatory of surface change for the NASA Space Apps Challenge 2026 ("Dancing with the SARs"). \
It reads radar from the NISAR satellite and rain from NASA, learns what is normal for each place, and projects radar and rain to 2030 with an honest error band. \
It covers South and Central America, focused on the Amazon.

DATA. NISAR (NASA and ISRO, L-band radar, 24 cm, 12-day repeat, since October 2025). OPERA RTC-S1 (Sentinel-1, C-band radar, 30 m, since 2016): the forecast learns from this one because NISAR is too young. \
GPM IMERG rain (0.1 degree cells, about 11 km). A radar pixel is a power value; we convert it to decibels, dB = 10*log10(x): dark means a weak echo (calm water), bright a strong one (buildings, wet soil). \
Clouds and night do not matter to radar, which is why it suits the Amazon.

MAP LAYERS (ids for the tool): nisar = NISAR backscatter (cyan); water = surface water, OPERA DSWx 30 m (blue); dist = vegetation loss alerts, OPERA DIST-ALERT 30 m (orange); \
imerg = rain rate, GPM IMERG (violet); lst = land surface temperature, VIIRS (red); smap = soil moisture, SMAP (green). Each layer keeps one color at every zoom. \
Air currents (wind) are animated streamlines of the 850 hPa wind, live from Open-Meteo. Base maps: satellite (Sentinel-2 cloudless), bluemarble, modis (daily true color). \
The timeline at the bottom runs 2016 to 2030: real data up to today, scenarios after it (drawn with stripes, never shown as real images). The 3D button tilts the view; buildings and streets show in cities.

THE ANALYSIS (machine learning, the core). For each place the model learns the yearly rhythm of the radar (VV and VH polarizations) and of the rain, month by month since 2016. \
It compares candidates (seasonal, seasonal + trend, + rain, + gradient boosting) by a rolling-origin backtest and keeps the best; the forecast has an 80 percent band that widens with the years ahead. \
It must beat a simple rule (the average of the same month in past years) on data it never saw, or the map shows the simple rule. Future rain is unknown, so the forecast assumes normal rain.

RISKS (screening signals, not predictions): vegetation loss (a lasting drop of cross-polarized radar), flood (past flood-like radar, recent rain against the place's climate), \
drought and fire (standardised 3-month rain and the 6-month forecast), unusual change (an Isolation Forest over monthly radar and rain). Levels: low below 0.3, moderate below 0.6, high from 0.6. \
A land-cover classifier (forest, other vegetation, city, open water) decides which risks make sense; on places and years it never saw it scored 85 percent with radar plus ground color, 73 percent with radar alone, against 54 percent for a simple rule, over 16 rounds. \
It did not reach the 90 percent target; more labeled places would help.

REGIONS. Three ready-made regional forecasts, built ahead from sample places: amazon, sao-paulo, rio-grande-do-sul (open from the Regions tab). For every risk the map shows the riskiest sample place. \
You can also click a point or drag an area (Shift+drag in 3D) and press the forecast button; the first forecast of a new place reads years of satellite scenes and takes minutes, the next ones are instant.

SIMULATIONS (illustrations of scale, never forecasts): earthquake, flood, wildfire, storm, placed on the map with growing damage zones and the buildings inside them.

AMAZON IMPACT. The Amazon exhales moisture and winds carry it south and west (flying rivers) to the Andes, the Centre-West and Southeast of Brazil including São Paulo, the South including Rio Grande do Sul, and the Plata basin. \
Studies link the forest to rain far downwind, but the Amazon is one important source among others (the Atlantic is another): say \"part of\", never \"all of\", and never claim it for a single storm.

WHAT THE NUMBERS MEAN. The 85 percent is how often the classifier recognised what a place IS (forest, other vegetation, city, water) on places and years it never saw. It is not the accuracy of the forecast. \
The forecast is judged by its backtest error and by how much better it is than repeating last year (shown in the Charts view as "vs last year").

HONEST WORDING. Say "the radar response changed", "a forecast has a band", "a scenario is simulated and uncertain". Never say a satellite detected deforestation, that a place will flood in a given year, or that the model is always right.

PLACES you can fly to, as (latitude, longitude): {places}. For any other place use your best knowledge of its coordinates.

WHERE THINGS ARE. The left panel (Layers) holds the base map, the six NASA layers with their opacity, the 3D view options and the natural-disaster simulations. The right panel (Inspector) has three tabs: Data (the radar and rain series of a point or area), Forecast (the verdict, then Risks, Places and Charts) and Regions (the three ready-made forecasts). The top bar has city shortcuts, search, the AI button and the language buttons. The timeline is at the bottom.

PAGES. The site has a presentation at "/" (the story), a satellite story at "/satelite", this map at "/mapa", and a technical page "Data and methods" at "/tecnico".\
"""

KNOWLEDGE = _KNOWLEDGE.format(places=PLACES)

SYSTEM = """\
You are the assistant inside the SARIS map. Answer in the language the person writes in (English, Portuguese or Spanish); if you cannot tell, use {language}. Keep it to about sixty words and at most four short sentences, for a non-specialist; be friendly and concrete. Pick the two or three most useful things instead of covering everything.

RULES
1. A QUESTION about the project, the data, the model, a risk, a number or "how does this work" gets a TEXT answer from the knowledge base. Do not call the tool for questions.
2. A REQUEST to change what the map shows (show or hide a layer, fly somewhere, open a region, run the forecast, simulate an event, move the year, change the base map, open a panel tab) gets the tool `control_map` and ONE short sentence saying what you did.
3. If someone asks where to find something ("where do I see the risks?"), explain in text where it is, and you may also open the right tab or region with the tool.
4. Numbers about a place come ONLY from the page state below. If the number is not there, say how to get it (open a region, or click a point and run the forecast). Never invent numbers.
5. A simulation (earthquake, flood, wildfire, storm) starts at the selected place, or at the centre of the view; if the person names a place, put its lat and lon in the action. Say it is an illustration of scale, not a forecast.
6. Prefer the ready-made regions for the Amazon, São Paulo and Rio Grande do Sul. For flying use zoom 3 for the continent, 6 for a country, 10 for a city, 13 for a neighbourhood.
7. If something is outside what the map can do, say so plainly. Never write your reasoning, only the answer.
8. Write the way a friendly person talks: plain flowing sentences in one short paragraph. No bullet points, no numbered lists, no headings, no bold, no emojis, no tables, and never a dash used as punctuation (no em dash, no en dash). Avoid stock phrases and cheerful openings like "Great question"; just answer.
9. Only talk about SARIS and its topics (radar, rain, the Amazon and South America, the data, the model, the map). For anything else, politely say you can only help with the map and the project.

KNOWLEDGE BASE
{knowledge}

CURRENT PAGE STATE
{state}
"""

TOOL = {
    "toolSpec": {
        "name": "control_map",
        "description": "Change what the map shows. Give the list of actions to run, in order.",
        "inputSchema": {"json": {
            "type": "object",
            "properties": {"actions": {"type": "array", "maxItems": MAX_ACTIONS, "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["layer", "allLayers", "wind", "region", "forecast", "fly", "disaster", "year", "base", "tab", "clear"],
                             "description": "layer: show/hide one NASA layer (needs id and on). allLayers: all on or off (needs on). wind: air currents (needs on). region: open a ready-made region (needs id). forecast: run the forecast of the selected point. fly: move the camera (needs lon, lat, zoom). disaster: simulate an event (needs kind; optional magnitude, rise, hours, wind, dryness, gust, and lat and lon to place it). year: move the timeline (needs year 2016-2030). base: base map (needs id). tab: open the panel tab (needs tab). clear: clear the selection and any simulation."},
                    "id": {"type": "string", "description": "layer: nisar, water, dist, imerg, lst, smap. region: amazon, sao-paulo, rio-grande-do-sul. base: satellite, bluemarble, modis."},
                    "on": {"type": "boolean"},
                    "lon": {"type": "number"}, "lat": {"type": "number"}, "zoom": {"type": "number"},
                    "kind": {"type": "string", "enum": sorted(KINDS)},
                    "magnitude": {"type": "number"}, "rise": {"type": "number"}, "hours": {"type": "number"}, "wind": {"type": "number"},
                    "dryness": {"type": "number"}, "gust": {"type": "number"},
                    "year": {"type": "integer"},
                    "tab": {"type": "string", "enum": sorted(TABS)},
                },
                "required": ["type"],
            }}},
            "required": ["actions"],
        }},
    }
}

DONE = {"en": "Done.", "pt": "Pronto.", "es": "Listo."}
LAYER_NAME = {
    "nisar": ("NISAR radar", "radar NISAR", "radar NISAR"), "water": ("surface water", "água superficial", "agua superficial"),
    "dist": ("vegetation loss alerts", "alertas de perda de vegetação", "alertas de pérdida de vegetación"), "imerg": ("rain", "chuva", "lluvia"),
    "lst": ("surface temperature", "temperatura da superfície", "temperatura de la superficie"), "smap": ("soil moisture", "umidade do solo", "humedad del suelo"),
}
REGION_NAME = {"amazon": "Amazon", "sao-paulo": "São Paulo", "rio-grande-do-sul": "Rio Grande do Sul"}
THINKING = re.compile(r"<thinking>.*?</thinking>|</?thinking>", re.S | re.I)


MD_LINE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", re.M)


def humanize(text: str) -> str:
    """The answers are meant to read like a person wrote them: plain sentences, no lists, no markdown, no travessão.
    The prompt asks for that; this makes sure, whatever the model does."""
    lines = []
    for ln in text.split("\n"):
        if MD_LINE.match(ln):                                  # a list item becomes a sentence of its own
            ln = MD_LINE.sub("", ln).strip()
            ln = ln[:1].upper() + ln[1:]
            if ln and ln[-1] not in ".!?:":
                ln += "."
        lines.append(ln)
    t = "\n".join(lines)
    t = re.sub(r"[*_`#]+", "", t)                              # bold, italics, code, headings
    t = re.sub(r"\s*[\u2014\u2013]\s*", ", ", t)                # travessões become a comma
    t = re.sub(r"\s*\n+\s*", " ", t)                           # one paragraph
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"([.!?:;])\s*,", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return re.sub(r"(?<=[a-z\u00e0-\u00ff0-9\)])\s+(?=[A-Z\u00c0-\u00dd][a-z\u00e0-\u00ff])", lambda m: m.group(0), t)


def strip_thinking(text: str) -> str:
    """Nova models sometimes print their reasoning in <thinking> tags; the person never needs to see it."""
    return re.sub(r"\n{3,}", "\n\n", THINKING.sub("", text)).strip()


def describe(actions: list[dict], lang: str) -> str:
    """One short sentence for what was done, in the person's language (used when the model wrote none)."""
    i = {"en": 0, "pt": 1, "es": 2}.get(lang, 0)
    parts: list[str] = []
    for a in actions:
        t = a["type"]
        if t == "layer":
            state = (("on", "ligada", "encendida"), ("off", "desligada", "apagada"))[0 if a["on"] else 1][i]
            parts.append(f"{LAYER_NAME[a['id']][i]} {state}")
        elif t == "wind":
            parts.append(("Air currents on", "Correntes de ar ligadas", "Corrientes de aire encendidas")[i] if a["on"] else ("Air currents off", "Correntes de ar desligadas", "Corrientes de aire apagadas")[i])
        elif t == "region":
            parts.append((f"Opening {REGION_NAME[a['id']]}", f"Abrindo {REGION_NAME[a['id']]}", f"Abriendo {REGION_NAME[a['id']]}")[i])
        elif t == "forecast":
            parts.append(("Running the forecast", "Gerando a previsão", "Generando el pronóstico")[i])
        elif t == "fly":
            parts.append(("Flying there", "Indo até lá", "Yendo allí")[i])
        elif t == "disaster":
            parts.append(("Simulation started", "Simulação iniciada", "Simulación iniciada")[i])
        elif t == "year":
            parts.append(str(a["year"]))
        elif t == "tab":
            parts.append(("Panel opened", "Painel aberto", "Panel abierto")[i])
        elif t == "base":
            parts.append(("Base map changed", "Mapa-base trocado", "Mapa base cambiado")[i])
    return ". ".join(dict.fromkeys(parts)) + "." if parts else DONE.get(lang, DONE["en"])


PT_HINT = re.compile(r"\b(n\u00e3o|voc\u00ea|uma|como|para|pra|com|mostra\w*|abr\w+|lig\w+|chuva|previs\u00e3o|obrigad\w*|por que|porque)\b|\u00e7\u00e3o|\u00f5es", re.I)
ES_HINT = re.compile(r"\b(c\u00f3mo|qu\u00e9|una|para|con|muestra\w*|abre|enciende|lluvia|pron\u00f3stico|gracias|por qu\u00e9|porque|est\u00e1)\b|\u00bf|\u00f1|ci\u00f3n", re.I)
EN_HINT = re.compile(r"\b(the|what|how|show|open|turn|why|where|and|is|are|please|rain|forecast)\b", re.I)


def guess_lang(question: str, default: str = "en") -> str:
    """The language the person wrote in (the page's language only breaks a tie), so a question in Portuguese gets Portuguese
    even when the interface is in English."""
    scores = {"pt": len(PT_HINT.findall(question)), "es": len(ES_HINT.findall(question)), "en": len(EN_HINT.findall(question))}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 and scores[best] >= scores.get(default, 0) else default


def configured() -> bool:
    return bool(env_value("BEDROCK_API_KEY"))


def model_id() -> str:
    return env_value("ASF_ASSISTANT_MODEL") or DEFAULT_MODEL


def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def sanitize(raw) -> list[dict]:
    """Keep only actions the page knows, with ids from the whitelists and numbers inside their ranges."""
    out: list[dict] = []
    for a in raw if isinstance(raw, list) else []:
        if not isinstance(a, dict) or len(out) >= MAX_ACTIONS:
            continue
        t = a.get("type")
        try:
            if t == "layer" and a.get("id") in LAYERS:
                out.append({"type": "layer", "id": a["id"], "on": bool(a.get("on", True))})
            elif t == "allLayers":
                out.append({"type": "allLayers", "on": bool(a.get("on", True))})
            elif t == "wind":
                out.append({"type": "wind", "on": bool(a.get("on", True))})
            elif t == "region" and a.get("id") in REGIONS:
                out.append({"type": "region", "id": a["id"]})
            elif t == "forecast":
                out.append({"type": "forecast"})
            elif t == "fly":
                out.append({"type": "fly", "lon": clamp(a["lon"], *LON_RANGE), "lat": clamp(a["lat"], *LAT_RANGE), "zoom": clamp(a.get("zoom", 8), 3, 15)})
            elif t == "disaster" and a.get("kind") in KINDS:
                params = {k: clamp(a[k], *PARAMS[k]) for k in PARAMS if k in a}
                act = {"type": "disaster", "kind": a["kind"], "params": params}
                if "lat" in a and "lon" in a:
                    act["lon"], act["lat"] = clamp(a["lon"], *LON_RANGE), clamp(a["lat"], *LAT_RANGE)
                out.append(act)
            elif t == "year":
                out.append({"type": "year", "year": int(clamp(a["year"], 2016, 2030))})
            elif t == "base" and a.get("id") in BASES:
                out.append({"type": "base", "id": a["id"]})
            elif t == "tab" and a.get("tab") in TABS:
                out.append({"type": "tab", "tab": a["tab"]})
            elif t == "clear":
                out.append({"type": "clear"})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def clean(question: str, state: str | None, history) -> tuple[str, str, list[dict]]:
    """Bound everything the page sends, so no one can make an enormous prompt."""
    q = (question or "").strip()[:MAX_QUESTION]
    st = (state or "").strip()[:MAX_CONTEXT]
    turns: list[dict] = []
    for h in (history or [])[-MAX_HISTORY:]:
        if isinstance(h, dict) and h.get("role") in ("user", "assistant") and isinstance(h.get("text"), str) and h["text"].strip():
            role = h["role"]
            if turns and turns[-1]["role"] == role:  # Converse needs alternating roles
                continue
            turns.append({"role": role, "content": [{"text": h["text"].strip()[:MAX_QUESTION]}]})
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    if turns and turns[-1]["role"] == "user":
        turns.pop()  # the new question is the user turn that follows
    return q, st, turns


def ask(question: str, lang: str, state: str | None, history=None, *, post=requests.post) -> dict:
    """Returns {answer, actions}. Raises ValueError for an empty question, requests errors for network trouble."""
    q, st, turns = clean(question, state, history)
    if not q:
        raise ValueError("empty question")
    lang = guess_lang(q, lang)
    key = env_value("BEDROCK_API_KEY")
    region = env_value("BEDROCK_REGION") or "us-east-1"
    body = {
        "system": [{"text": SYSTEM.format(language=LANGS.get(lang, "English"), knowledge=KNOWLEDGE, state=st or "nothing selected")}],
        "messages": [*turns, {"role": "user", "content": [{"text": q}]}],
        "inferenceConfig": {"maxTokens": 350, "temperature": 0.3},
        "toolConfig": {"tools": [TOOL], "toolChoice": {"auto": {}}},
    }
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{quote(model_id(), safe='')}/converse"
    res = post(url, json=body, timeout=30, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    res.raise_for_status()
    blocks = res.json().get("output", {}).get("message", {}).get("content", [])
    text = humanize(strip_thinking(" ".join(b["text"].strip() for b in blocks if "text" in b)))
    actions: list[dict] = []
    for b in blocks:
        use = b.get("toolUse")
        if use and use.get("name") == "control_map":
            actions += sanitize((use.get("input") or {}).get("actions"))
    actions = actions[:MAX_ACTIONS]
    if not text and actions:
        text = _follow_up(post, url, key, body, blocks, actions, lang) or describe(actions, lang)
    return {"answer": text, "actions": actions}


def _follow_up(post, url: str, key: str, body: dict, blocks: list, actions: list[dict], lang: str) -> str:
    """The model called the tool and wrote nothing: give it the tool result so it writes the sentence for the person
    (an answer if they asked a question, a confirmation if they gave an order). Failure just means the fixed sentence."""
    use = next((b["toolUse"] for b in blocks if b.get("toolUse")), None)
    if not use:
        return ""
    done = json.dumps(actions, ensure_ascii=False)
    messages = [*body["messages"], {"role": "assistant", "content": [{"toolUse": use}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": use["toolUseId"], "content": [{"text": f"The page will now run: {done}. Now write your short answer for the person, in their language; if they asked a question, answer it."}]}}]}]
    try:
        res = post(url, json={**body, "messages": messages, "inferenceConfig": {"maxTokens": 400, "temperature": 0.2}}, timeout=30,
                   headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        res.raise_for_status()
        out = res.json().get("output", {}).get("message", {}).get("content", [])
        return humanize(strip_thinking(" ".join(b["text"] for b in out if "text" in b)))
    except Exception:
        return ""


class RateLimit:
    """A few requests a minute per client: the key is shared by everyone who opens the page."""

    def __init__(self, per_minute: int = 20) -> None:
        self.per_minute = per_minute
        self.seen: dict[str, deque] = defaultdict(deque)

    def allow(self, who: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        q = self.seen[who]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True
