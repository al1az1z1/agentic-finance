from __future__ import annotations

def choose_agents(has_news: bool, has_prices: bool, has_technicals: bool, has_earnings: bool) -> list[str]:
    agents = []
    if has_news:
        agents.append("news")
    if has_technicals and has_prices:
        agents.append("technical")
    if has_earnings:
        agents.append("earnings")
    agents.append("risk")
    return agents

