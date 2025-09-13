import textwrap
from datetime import timedelta, date
import random

USE_GEMINI = False
try:
    import google.generativeai as genai

    # ✅ Hardcoded API key (demo) – better: use environment variable
    GEMINI_API_KEY = "AIzaSyAEVPVVGPoTV89Rl20sVQuYs7f8rBoHLXQ"
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        USE_GEMINI = True
except Exception:
    USE_GEMINI = False


def generate_itinerary_llm(destination: str, country: str, days: int, start_date: date,
                           interests: list, budget_cat: str, travellers: int, pace: str) -> str:
    # Convert budget from USD to INR (if budget_cat is in USD)
    try:
        if "$" in budget_cat or "USD" in budget_cat:
            usd_value = float(budget_cat.replace("USD", "").replace("$", "").replace("/day", "").strip())
            inr_value = int(usd_value * 83)  # Example conversion rate 1 USD = 83 INR
            budget_cat_inr = f"₹{inr_value}/day"
        else:
            budget_cat_inr = budget_cat
    except:
        budget_cat_inr = budget_cat

    # ✨ Random element to make output unique each time
    creativity = random.choice([
        "Make it food-centric with hidden gems.",
        "Add some cultural festivals or local experiences.",
        "Include some offbeat adventure activities.",
        "Balance sightseeing with relaxation spots.",
        "Highlight famous cafes, nightlife and local shopping."
    ])

    system_prompt = f"""
You are a meticulous AI travel planner.
Build a UNIQUE and CREATIVE day-by-day itinerary for {destination}, {country}.

Constraints:
- Trip length: {days} days starting {start_date.isoformat()}.
- Interests: {', '.join(interests) if interests else 'general sightseeing'}.
- Budget: {budget_cat_inr}.
- Travellers: {travellers}.
- Travel pace: {pace}.
- Important: Every time generate a different itinerary with variety. {creativity}
    """.strip()

    if USE_GEMINI:
        try:
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            resp = model.generate_content(system_prompt)
            if resp and resp.text.strip():
                return resp.text  # ✅ Always AI-based
        except Exception as e:
            print("Gemini error:", e)

    # ⚠️ fallback only if Gemini not working at all
    blocks = []
    for i in range(days):
        d = start_date + timedelta(days=i)
        blocks.append(textwrap.dedent(f"""
        Day {i+1} – {d.strftime('%A, %d %b %Y')}
        • Explore local attractions  
        • Enjoy food experiences  
        • Evening walk & nightlife  
        """))
    return f"## {destination}, {country} — {days} Days\n" + "\n".join(blocks)
