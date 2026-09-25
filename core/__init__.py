"""
core package — Central configuration, database state, risk engine, and AI gatekeeper.
"""

from .config import (
    TradingConfig,
    AccountConfig,
    RiskConfig,
    InstrumentConfig,
    TimeframeConfig,
    PairSettings,
    Direction,
    MarketBias,
    StrategyType,
    DEFAULT_CONFIG,
    get_instrument,
    normalize_strategy_key,
)
from .state import (
    StateManager,
    TradeRecord,
    normalize_strategy_display_name,
    format_duration,
)
from .risk_engine import RiskEngine, AuthorizationResult
from .conflict_resolver import ConflictResolver
from .ai_analyst import AIAnalyst, AIDecision
from .market_regime import MarketRegimeDetector, MarketRegime, RegimeAnalysis
from .news_filter import NewsFilter

__all__ = [
    "TradingConfig",
    "AccountConfig",
    "RiskConfig",
    "InstrumentConfig",
    "TimeframeConfig",
    "PairSettings",
    "Direction",
    "MarketBias",
    "StrategyType",
    "DEFAULT_CONFIG",
    "get_instrument",
    "normalize_strategy_key",
    "normalize_strategy_display_name",
    "format_duration",
    "StateManager",
    "TradeRecord",
    "RiskEngine",
    "AuthorizationResult",
    "ConflictResolver",
    "AIAnalyst",
    "AIDecision",
    "MarketRegimeDetector",
    "MarketRegime",
    "RegimeAnalysis",
    "NewsFilter",
]
