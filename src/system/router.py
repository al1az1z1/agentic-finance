from __future__ import annotations

def choose_agents(has_news: bool, has_prices: bool, has_technicals: bool) -> list[str]:
    agents = []
    if has_news: agents.append("news")
    # earnings optional if you add a financials fetch later
    if has_technicals and has_prices: agents.append("technical")
    agents.append("risk")
    return agents
