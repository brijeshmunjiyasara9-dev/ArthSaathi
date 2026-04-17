"""
chat_service.py — Multi-step conversation logic for ArthSaathi.

The bot collects user information across 6 steps, then runs ML models
and generates personalized advice via OpenAI.
"""
from typing import Dict, List, Any, Tuple

# Step prompts for structured collection
STEP_PROMPTS = {
    1: (
        "Namaste! 🙏 I'm **ArthSaathi** — your personal financial wellness advisor.\n\n"
        "I'll help assess your household's financial health and give you personalized advice. "
        "This will take just 5–6 minutes.\n\n"
        "Let's start! **Which state do you live in?** (e.g., Maharashtra, Gujarat, Tamil Nadu, Delhi...)"
    ),
    2: (
        "Thank you! Now, **how many family members** live in your household? "
        "(Include everyone — children, elderly, etc.)"
    ),
    3: (
        "Got it! Is your household in an **Urban or Rural** area?"
    ),
    4: (
        "Now let's talk about income. 💰\n"
        "What is your **total household monthly income** (from all sources combined)? "
        "Please give an approximate amount in ₹."
    ),
    5: (
        "What are your **total monthly household expenses**? "
        "(Include food, rent, EMIs, bills, everything. Approximate amount in ₹ is fine.)"
    ),
    6: (
        "How much does your family spend on **food per month**? (₹)"
    ),
    7: (
        "Do you have any **EMI or loan payments** every month? "
        "If yes, what is the total monthly EMI amount? (Enter 0 if none)"
    ),
    8: (
        "What are your approximate **monthly health expenses**? "
        "(Medicines, doctor visits, hospital — enter 0 if none)"
    ),
    9: (
        "Is any family member currently **hospitalised**, or on **regular long-term medication**? "
        "(Reply: 'neither', 'hospitalised', 'medication', or 'both')"
    ),
    10: (
        "Does your family have **health insurance**? (Yes/No)"
    ),
    11: (
        "Does anyone in your family have a **bank account**? (Yes/No)"
    ),
    12: (
        "Does anyone have **life insurance** (LIC or any policy)? (Yes/No)"
    ),
    13: (
        "What is the **primary earner's occupation**? "
        "(e.g., Salaried, Self-employed, Farmer, Business owner, Daily wage worker, Retired)"
    ),
    14: (
        "What is the **highest education level** in your household? "
        "(e.g., Below Primary, Primary, Secondary, Graduate, Post Graduate)"
    ),
    15: (
        "Almost done! 🎉 What is the approximate **age of the head of your household**?"
    ),
}

TOTAL_STEPS = 15


def _parse_bool(text: str) -> bool:
    t = text.lower().strip()
    return t in ('yes', 'y', 'ha', 'haan', 'true', '1', 'ji han', 'bilkul')


def _parse_number(text: str) -> float:
    """Extract first number (possibly with commas/K/L) from text."""
    import re
    text = text.replace(',', '').replace('₹', '').replace('rupees', '').strip()
    # Handle lakh/K notation
    m = re.search(r'(\d+\.?\d*)\s*(lakh|l\b)', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 100_000
    m = re.search(r'(\d+\.?\d*)\s*k\b', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1_000
    m = re.search(r'\d+\.?\d*', text)
    return float(m.group()) if m else 0.0


def process_user_response(step: int, user_text: str,
                           profile: Dict[str, Any]) -> Dict[str, Any]:
    """Parse user answer for the given step and update profile dict."""
    t = user_text.strip()

    if step == 1:
        profile['state'] = t.title()
    elif step == 2:
        profile['num_members'] = max(1, int(_parse_number(t) or 1))
    elif step == 3:
        region = t.lower()
        profile['region_type'] = 'Urban' if 'urban' in region or 'city' in region or 'metro' in region else 'Rural'
    elif step == 4:
        profile['monthly_income'] = _parse_number(t)
    elif step == 5:
        profile['monthly_total_expense'] = _parse_number(t)
    elif step == 6:
        profile['monthly_food_expense'] = _parse_number(t)
    elif step == 7:
        profile['monthly_emi'] = _parse_number(t)
    elif step == 8:
        profile['monthly_health_expense'] = _parse_number(t)
    elif step == 9:
        tl = t.lower()
        profile['is_hospitalised'] = 'hospitalised' in tl or 'hospital' in tl or tl == 'both'
        profile['is_on_medication'] = 'medication' in tl or 'medicine' in tl or tl == 'both'
    elif step == 10:
        profile['has_health_insurance'] = _parse_bool(t)
    elif step == 11:
        profile['has_bank_account'] = _parse_bool(t)
    elif step == 12:
        profile['has_life_insurance'] = _parse_bool(t)
    elif step == 13:
        profile['occupation'] = t.title()
    elif step == 14:
        profile['education'] = t.title()
    elif step == 15:
        profile['age_head'] = max(18, min(90, int(_parse_number(t) or 40)))

    return profile


def get_next_step_prompt(step: int, profile: Dict[str, Any]) -> str:
    """Return the bot's question for the given step, with context."""
    prompt = STEP_PROMPTS.get(step, "")

    # Personalize with name/state
    if step == 2 and profile.get('state'):
        prompt = f"Great, {profile['state']}! 🌏 " + prompt

    return prompt


def format_profile_summary(profile: Dict[str, Any]) -> str:
    """Create a readable summary of collected profile."""
    lines = [
        f"**State:** {profile.get('state', 'N/A')}",
        f"**Area:** {profile.get('region_type', 'N/A')}",
        f"**Family size:** {profile.get('num_members', 'N/A')} members",
        f"**Monthly income:** ₹{profile.get('monthly_income', 0):,.0f}",
        f"**Monthly expenses:** ₹{profile.get('monthly_total_expense', 0):,.0f}",
        f"**Food expenses:** ₹{profile.get('monthly_food_expense', 0):,.0f}",
        f"**EMI payments:** ₹{profile.get('monthly_emi', 0):,.0f}",
        f"**Health expenses:** ₹{profile.get('monthly_health_expense', 0):,.0f}",
    ]
    return "\n".join(lines)
