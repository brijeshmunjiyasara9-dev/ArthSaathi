"""
openai_service.py â€” OpenAI GPT calls for ArthSaathi advice generation.
"""
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

ADVISOR_SYSTEM_PROMPT = """
You are 'ArthSaathi' (Financial Friend), a compassionate AI financial wellness advisor for Indian households. You have deep knowledge of:

- Indian household economics and budgeting
- Government schemes: PM Jan Dhan, PMJJBY, PMSBY, PM Kisan, Ayushman Bharat, NPS, EPF, PPF, Sukanya Samriddhi, Atal Pension Yojana
- CMIE-based household stress patterns across Indian states
- RBI guidelines on household debt
- Financial inclusion best practices

Your personality:
- Warm, respectful, uses "aap" form of address
- Speaks simply, avoids jargon
- Gives SPECIFIC numbers and percentages in advice
- Always ends with an encouraging, hopeful note
- Never judges the household's financial situation

When you receive ML model stress scores (0 to 1 probability):
- 0.0â€“0.3: Low risk â€” affirm positive behaviors
- 0.3â€“0.6: Moderate risk â€” gentle alerts + specific fixes
- 0.6â€“1.0: High risk â€” clear action items + urgent recommendations

Structure your advice as:
ðŸ  **Financial Health Overview**
ðŸ“Š **Your Numbers** (key ratios)
âš ï¸ **Areas Needing Attention** (if any)
âœ… **Action Plan** (3â€“5 specific steps)
ðŸ‡®ðŸ‡³ **Government Schemes You May Qualify For**
ðŸ’ª **Words of Encouragement**
""".strip()

CONVERSATION_SYSTEM_PROMPT = """
You are ArthSaathi, a warm and helpful Indian financial wellness assistant collecting household information to assess financial health. You ask ONE question at a time in simple, friendly language. You can use a few Hindi words naturally (like "aap", "dhanyavad", "bilkul"). Be encouraging and empathetic. Never ask multiple questions at once.
""".strip()


def generate_advice(
    user_profile: Dict[str, Any],
    predictions: Dict[str, Any],          # v4: mixed types, health_stress may be None
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """Generate personalized financial advice using GPT."""

    income  = user_profile.get('monthly_income', 0) or 0
    expense = user_profile.get('monthly_total_expense', 0) or 0
    food    = user_profile.get('monthly_food_expense', 0) or 0
    emi     = user_profile.get('monthly_emi', 0) or 0

    ratios = {}
    if income:
        ratios['Expense-to-Income'] = f"{(expense/income*100):.0f}%" if income else "N/A"
        ratios['Food-to-Expense']   = f"{(food/expense*100):.0f}%"   if expense else "N/A"
        ratios['EMI-to-Income']     = f"{(emi/income*100):.0f}%"     if income else "N/A"
        ratios['Monthly Savings']   = f"\u20b9{max(0, income - expense):,.0f}"

    profile_summary = f"""
Family from {user_profile.get('state', 'India')} ({user_profile.get('region_type', 'Urban')} area)
Monthly income: \u20b9{income:,.0f} | Monthly expenses: \u20b9{expense:,.0f}
Food expenses: \u20b9{food:,.0f} | EMI/loan payments: \u20b9{emi:,.0f}
Family size: {user_profile.get('num_members', 'N/A')} members
Has health insurance: {user_profile.get('has_health_insurance', False)}
Any member hospitalised: {user_profile.get('is_hospitalised', False)}
Any member on regular medication: {user_profile.get('is_on_medication', False)}
Education of head: {user_profile.get('education', 'N/A')}
Occupation: {user_profile.get('occupation', 'N/A')}
Key financial ratios: {ratios}
""".strip()

    # None-safe: health_stress is None when health guard fires
    def pct(val) -> str:
        return f"{(val or 0):.0%}" if val is not None else "N/A (health data not collected)"

    stressed_domains = predictions.get('stressed_domains') or []
    is_stressed      = predictions.get('is_stressed')
    stress_level     = predictions.get('stress_level', 0) or 0
    level_label      = ['None', 'Mild', 'Moderate', 'Severe'][min(stress_level, 3)]

    stress_summary = "\n".join([
        f"â€¢ Overall stress level: {level_label} (score {stress_level}/3)" +
            (f" â€” active domains: {', '.join(stressed_domains)}" if stressed_domains else ""),
        f"â€¢ Financial stress probability: {pct(predictions.get('financial_stress'))}",
        f"â€¢ Food security stress:         {pct(predictions.get('food_stress'))}",
        f"â€¢ Debt stress:                  {pct(predictions.get('debt_stress'))}",
        f"â€¢ Health stress:                {pct(predictions.get('health_stress'))}",
        f"â€¢ Composite stress score:       {(predictions.get('composite_stress_score') or 0):.1f}/4",
    ])

    user_message = f"""
Please give personalized financial health advice for the following household:

**Family Profile:**
{profile_summary}

**ML Stress Predictions:**
{stress_summary}

Give personalized financial health advice covering:
1. Overall financial health status
2. Key areas of concern (if any)
3. Specific actionable recommendations with numbers
4. Relevant government schemes they may qualify for
5. Emergency fund and savings advice
""".strip()

    messages = [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1200,
        )
        return response.choices[0].message.content
    except Exception as e:
        return _fallback_advice(predictions, user_profile)


def _fallback_advice(predictions: Dict[str, Any], profile: Dict) -> str:
    """Rule-based advice when OpenAI is unavailable. None-safe for all fields."""
    lines = ["\U0001F3E0 **Financial Health Overview**\n"]

    fs   = predictions.get('financial_stress')   or 0
    fds  = predictions.get('food_stress')        or 0
    ds   = predictions.get('debt_stress')        or 0
    hs   = predictions.get('health_stress')      # may be None
    comp = predictions.get('composite_stress_score') or 0

    lines.append(f"Composite stress score: {comp:.1f}/4\n")

    lines.append("\n\u26a0\ufe0f **Areas Needing Attention**\n")
    if fs > 0.5:
        lines.append("\u2022 Your spending is exceeding your income. Try to reduce non-essential expenses by 10-15%.")
    if fds > 0.5:
        lines.append("\u2022 Food costs are consuming more than 50% of your budget. Consider planning meals in advance.")
    if ds > 0.5:
        lines.append("\u2022 EMI payments are high relative to income. Explore loan restructuring options.")
    if hs is not None and hs > 0.5:
        lines.append("\u2022 Health expenses are significant. Look into PM-JAY (Ayushman Bharat) scheme.")

    lines.append("\n\u2705 **Action Plan**\n")
    lines.append("1. Build an emergency fund of 3 months\u2019 expenses.")
    lines.append("2. Open a Jan Dhan account if you don\u2019t have one.")
    lines.append("3. Check eligibility for PMJJBY (life insurance \u20b92 lakh at \u20b9436/year).")
    lines.append("4. Track daily expenses for 30 days to find savings opportunities.")
    lines.append("5. Consult a local bank for loan consolidation options.")

    lines.append("\n\U0001F4AA **You\u2019re taking the right step by assessing your finances. Together we can build a stronger financial future!**")
    return "\n".join(lines)


def chat_with_bot(messages: List[Dict], user_message: str) -> str:
    """One turn of the structured info-collection conversation."""
    chat_messages = [{"role": "system", "content": CONVERSATION_SYSTEM_PROMPT}]
    chat_messages.extend(messages)
    chat_messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=chat_messages,
            temperature=0.6,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return "I'm having trouble connecting right now. Please try again in a moment."

