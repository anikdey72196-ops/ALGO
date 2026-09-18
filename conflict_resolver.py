from __future__ import annotations
from dataclasses import dataclass
from loguru import logger
from config import RiskConfig, MarketBias, Direction, normalize_strategy_key
from strategy import TradeSignal, LTFConfirmation

@dataclass(frozen=True)
class FilterResult:
    """Result of signal filtering with rejection tracking."""
    accepted_signal: TradeSignal | None
    total_signals: int
    rejected_count: int
    rejection_reasons: list[str]  # List of human-readable rejection reasons
    accepted_signals: tuple[TradeSignal, ...] = ()


class ConflictResolver:
    """Resolves conflicting signals and enforces quality gates."""
    
    def __init__(self, risk_config: RiskConfig):
        self.risk_config = risk_config
    
    def resolve(
        self,
        signals: list[TradeSignal],
        current_spread: float,
    ) -> FilterResult:
        """
        Filter and rank signals through 4 quality gates:
        
        Gate 1 — HTF Alignment:
          Reject any signal where direction opposes HTF bias.
          BUY is valid only if htf_bias == BULLISH.
          SELL is valid only if htf_bias == BEARISH.
          If htf_bias == NEUTRAL, reject all.
        
        Gate 2 — Minimum R:R:
          Reject if rr_ratio < min_rr (1.4 for 5M scalps, risk_config.min_rr_ratio for swing).
        
        Gate 3 — Spread Viability:
          Reject if tp_distance < risk_config.min_tp_spread_multiple * current_spread.
          (TP must be at least 6× the spread to cover friction.)
        
        Gate 4 — Rank by quality_score, return the single best signal.
        
        Returns FilterResult with the accepted signal (or None) and all rejection reasons.
        """
        if not signals:
            return FilterResult(
                accepted_signal=None,
                total_signals=0,
                rejected_count=0,
                rejection_reasons=[]
            )
            
        rejection_reasons = []
        survivors = []
        
        for signal in signals:
            # Gate 1: HTF Alignment
            if signal.htf_bias == MarketBias.NEUTRAL or \
               (signal.direction == Direction.BUY and signal.htf_bias != MarketBias.BULLISH) or \
               (signal.direction == Direction.SELL and signal.htf_bias != MarketBias.BEARISH):
                reason = f"[HTF_CONFLICT] {signal.symbol} {signal.direction.value} rejected: opposes HTF bias ({signal.htf_bias.value})"
                logger.info(reason)
                rejection_reasons.append(reason)
                continue
                
            # Gate 2: Minimum R:R (Adaptive: 1.4 for 5M OB scalps, 1.8 for ICT setups, standard min_rr_ratio for swing)
            conf_val = getattr(signal.ltf_confirmation, 'value', str(signal.ltf_confirmation))
            strat_val = getattr(signal, 'strategy_name', '') or ''
            if signal.ltf_confirmation == LTFConfirmation.OB_SCALP_5M or "SCALP" in strat_val.upper():
                min_rr = 1.4
            elif conf_val.startswith("ICT_") or "ICT" in strat_val.upper():
                min_rr = 1.8
            else:
                min_rr = self.risk_config.min_rr_ratio

            if signal.rr_ratio < min_rr:
                reason = f"[LOW_RR] {signal.symbol} R:R {signal.rr_ratio:.2f} < minimum {min_rr}"
                logger.info(reason)
                rejection_reasons.append(reason)
                continue
                
            # Gate 3: Spread Viability
            if signal.tp_distance < self.risk_config.min_tp_spread_multiple * current_spread:
                reason = f"[SPREAD_FRICTION] {signal.symbol} TP distance {signal.tp_distance:.5f} < {self.risk_config.min_tp_spread_multiple}x spread ({current_spread:.5f})"
                logger.info(reason)
                rejection_reasons.append(reason)
                continue
                
            survivors.append(signal)
            
        total_signals = len(signals)
        rejected_count = len(rejection_reasons)
        
        if not survivors:
            logger.info("No signals passed quality gates")
            return FilterResult(
                accepted_signal=None,
                total_signals=total_signals,
                rejected_count=rejected_count,
                rejection_reasons=rejection_reasons
            )
            
        # Gate 4: Rank by quality_score descending
        survivors.sort(key=lambda s: s.quality_score, reverse=True)
        best_signal = survivors[0]

        # Deduplicate to keep at most 1 best signal per strategy, ordered by quality_score
        seen_strats = set()
        accepted_per_strat: list[TradeSignal] = []
        for s in survivors:
            strat_key = normalize_strategy_key(getattr(s, 'strategy_id', None) or s.strategy_name, getattr(s, 'magic_number', None))
            if strat_key not in seen_strats:
                seen_strats.add(strat_key)
                accepted_per_strat.append(s)

        logger.info(f"[ACCEPTED] {best_signal.symbol} {best_signal.direction.value} | R:R={best_signal.rr_ratio:.2f} | Score={best_signal.quality_score:.1f} | {best_signal.ltf_confirmation.value}")

        return FilterResult(
            accepted_signal=best_signal,
            total_signals=total_signals,
            rejected_count=rejected_count,
            rejection_reasons=rejection_reasons,
            accepted_signals=tuple(accepted_per_strat),
        )
