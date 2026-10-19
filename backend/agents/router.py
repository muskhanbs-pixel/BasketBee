"""
BasketBee - intent router (the Supervisor's first decision).

Two paths:
  FAST PATH  - keyword rules recognise the common questions instantly (no LLM call).
               On a laptop CPU every LLM call costs 30-60 s, so this matters a lot.
  SLOW PATH  - if no rule matches, the LLM classifies the question (one short call).

Possible plans:
    ["forecast"]               demand questions
    ["inventory"]              stock / run-out / reorder questions
    ["forecast", "inventory"]  compound questions ("how much will we sell AND do we have enough?")
    ["smalltalk"]              greetings, "what can you do?"
    ["coming_soon"]            transport / supplier / warehouse agents (designed, built after Review II)
    ["out_of_scope"]           anything else
"""

import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

GREETING = re.compile(r"^\s*(hi|hello|hey|namaste|good (morning|afternoon|evening)|thanks|thank you)\b"
                      r"|what can you do|how can you help|^\s*help\s*\??\s*$", re.I)
FORECAST = re.compile(r"\b(forecasts?|predict\w*|projection|demand|expected sales|"
                      r"how (many|much)\b.*\b(sell|sold|need)|will\b.*\bsell|sales (next|for|in|this))\b", re.I)
INVENTORY = re.compile(r"\b(run(ning)? out|runs out|out of stock|stock\w*|inventory|re-?order\w*|replenish\w*|"
                       r"shortage|low on|enough|purchase orders?|at risk|do we have|how (many|much)\b.*\b(have|left))\b",
                       re.I)
FESTIVAL_PREP = re.compile(r"\b(diwali|festival|ganesh|ugadi|sankranti|christmas)\b.*\b(ready|prepare\w*|plan\w*)\b"
                           r"|\b(ready|prepare\w*|plan\w*)\b.*\b(diwali|festival|ganesh|ugadi|sankranti|christmas)\b",
                           re.I)
COMING_SOON = re.compile(r"\b(route|routes|routing|deliver\w*|vehicles?|trucks?|shipments?|fleet|"
                         r"supplier performance|best supplier|warehouse capacity|utili[sz]ation)\b", re.I)

INTENT_TO_PLAN = {
    "forecast": ["forecast"], "inventory": ["inventory"], "both": ["forecast", "inventory"],
    "smalltalk": ["smalltalk"], "coming_soon": ["coming_soon"], "out_of_scope": ["out_of_scope"],
}

CLASSIFIER_PROMPT = """You route questions for BasketBee, a grocery distributor's supply-chain assistant.
Reply with exactly ONE word from this list:
forecast      - predicting future demand or sales of products
inventory     - current stock, what will run out, reorders, shortages
both          - needs a demand forecast AND a stock check
smalltalk     - greetings or questions about what you can do
coming_soon   - delivery routes, vehicles, shipments, supplier performance, warehouse capacity
out_of_scope  - anything unrelated to BasketBee's supply chain"""


@dataclass
class Route:
    intent: str
    plan: list[str]
    method: str        # "rules" | "llm" | "default"


def route_by_rules(question: str) -> Route | None:
    q = question.strip()
    f, i = bool(FORECAST.search(q)), bool(INVENTORY.search(q))
    if f and i:
        return Route("both", INTENT_TO_PLAN["both"], "rules")
    if i or FESTIVAL_PREP.search(q):
        return Route("inventory", INTENT_TO_PLAN["inventory"], "rules")
    if f:
        return Route("forecast", INTENT_TO_PLAN["forecast"], "rules")
    if COMING_SOON.search(q):
        return Route("coming_soon", INTENT_TO_PLAN["coming_soon"], "rules")
    if GREETING.search(q):
        return Route("smalltalk", INTENT_TO_PLAN["smalltalk"], "rules")
    return None


def route_by_llm(question: str, llm: BaseChatModel) -> Route:
    try:
        reply = llm.invoke([SystemMessage(CLASSIFIER_PROMPT), HumanMessage(question)]).content
    except Exception:   # noqa: BLE001 - LLM down: fall back safely
        return Route("out_of_scope", INTENT_TO_PLAN["out_of_scope"], "default")
    words = re.findall(r"[a-z_]+", (reply or "").lower())
    intent = next((w for w in words if w in INTENT_TO_PLAN), "out_of_scope")
    return Route(intent, INTENT_TO_PLAN[intent], "llm")


def route(question: str, llm: BaseChatModel | None = None) -> Route:
    return route_by_rules(question) or (route_by_llm(question, llm) if llm else
                                        Route("out_of_scope", INTENT_TO_PLAN["out_of_scope"], "default"))