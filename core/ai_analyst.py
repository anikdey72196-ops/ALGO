"""

ai_analyst.py — AI Second-Opinion Confirmation Gatekeeper using Google Gemini.



Features:

- Evaluates candidate trade setups using Google Gemini (gemini-2.0-flash / gemini-1.5-flash).

- Checks for retail stop-hunts, fakeout risks, momentum divergences, and macro alignment.

- Returns a structured JSON verdict: { confirmed: bool, confidence: float, reason: str }.

- Includes smart offline heuristic fallback if GEMINI_API_KEY is not yet supplied.

"""



from __future__ import annotations



import os

import json

from dataclasses import dataclass

from loguru import logger



from typing import TYPE_CHECKING

from core.config import TradingConfig, MarketBias, Direction



if TYPE_CHECKING:

    from strategies.strategy import TradeSignal, HTFAnalysis





@dataclass(frozen=True)

class AIDecision:

    """AI Trade Evaluation Verdict."""

    confirmed: bool

    confidence: float

    reason: str

    warnings: str = ""





class AIAnalyst:

    """Institutional Order Flow AI Gatekeeper powered by Gemini."""



    def __init__(self, config: TradingConfig):

        self.config = config

        self.model_name = getattr(config, "gemini_model", None) or os.environ.get("GEMINI_MODEL") or "gemini-flash-lite-latest"

        self.threshold = config.ai_confidence_threshold



        # Collect and deduplicate all configured API keys (GEMINI_API_KEYS, GEMINI_API_KEY, GOOGLE_API_KEY)

        raw_keys: list[str] = []

        if getattr(config, "gemini_api_key", None):

            raw_keys.append(config.gemini_api_key.strip())

        for env_var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY"):

            val = os.environ.get(env_var, "").strip()

            if val:

                for k in val.split(","):

                    k = k.strip()

                    if k and k not in raw_keys:

                        raw_keys.append(k)



        self.api_keys = raw_keys

        self._current_key_idx = 0

        self.api_key = self.api_keys[0] if self.api_keys else None

        self._clients: dict[str, Any] = {}



        if self.api_keys:

            try:

                from google import genai

                for k in self.api_keys:

                    self._clients[k] = genai.Client(api_key=k)

                masked_keys = [f"{k[:6]}...{k[-4:]}" for k in self.api_keys]

                logger.info(f"AIAnalyst initialized with {len(self.api_keys)} Gemini key(s) in rotation pool {masked_keys} ({self.model_name}).")

            except Exception as e:

                logger.warning(f"Could not initialize Google GenAI Client: {e}. Running in smart fallback mode.")

        else:

            logger.info("No GEMINI_API_KEY / GOOGLE_API_KEY provided. AIAnalyst will run high-conviction heuristic AI analysis.")



    @property

    def _client(self):

        """Active GenAI client from the key pool."""

        if not self.api_keys:

            return None

        current_k = self.api_keys[self._current_key_idx % len(self.api_keys)]

        return self._clients.get(current_k, getattr(self, '_custom_client', None))

    @_client.setter
    def _client(self, val):
        self._custom_client = val
        if val is not None and not self.api_keys:
            self.api_keys = ['custom_key']
            self._clients['custom_key'] = val



    def evaluate_setup(

        self,

        signal: TradeSignal,

        htf_analysis: HTFAnalysis,

        current_spread: float,

    ) -> AIDecision:

        """

        Evaluate candidate setup and return AIDecision.

        If AI confirmation is disabled in config, auto-confirms with 100% confidence.

        """

        if not self.config.ai_confirmation_enabled:

            return AIDecision(

                confirmed=True,

                confidence=100.0,

                reason="AI Confirmation Gate is bypassed in config.",

            )



        # 1. Prepare structured market summary prompt

        prompt = self._build_evaluation_prompt(signal, htf_analysis, current_spread)



        # 2. Query Gemini API if client is available

        if self._client:

            try:

                return self._call_gemini_api(prompt)

            except Exception as err:

                logger.error(f"Gemini API request failed: {err}. Falling back to internal heuristic audit.")



        # 3. Smart Heuristic AI Fallback (Validates liquidity sweeps & score thresholds)

        return self._heuristic_evaluation(signal, htf_analysis, current_spread)



    def _build_evaluation_prompt(

        self,

        signal: TradeSignal,

        htf_analysis: HTFAnalysis,

        current_spread: float,

    ) -> str:

        """Construct a rigorous prompt for institutional order flow analysis."""

        return f"""

You are an expert Institutional Quantitative Trading Analyst & Risk Controller.

Evaluate the following trade setup for execution quality and trap/fakeout risk.



Market & Trade Details:

- Instrument: {signal.symbol}

- Side: {signal.direction.value}

- Entry Price: {signal.entry_price:.5f}

- Stop Loss: {signal.stop_loss:.5f} (Distance: {signal.sl_distance:.5f})

- Take Profit: {signal.take_profit:.5f} (Distance: {signal.tp_distance:.5f})

- Risk-to-Reward Ratio: {signal.rr_ratio:.2f}:1

- Current Spread: {current_spread:.5f}



Smart Money Structure:

- Higher Timeframe Bias (1H): {signal.htf_bias.value} (Trend score: {htf_analysis.trend_clarity_score}/30)

- 1H 200 EMA Level: {htf_analysis.ema_value:.5f}

- Execution Setup Type: {signal.ltf_confirmation.value}

- Quality Score: {signal.quality_score:.1f}/100



Decision Rules:

1. Approve ONLY if the setup aligns with institutional order flow, has a clean invalidation point, and reward justifies risk.

2. Reject if there is high risk of a liquidity hunt continuation, counter-trend fakeout, or low conviction.

3. Return response in STRICT JSON format:

{{

  "confirmed": true or false,

  "confidence": float between 0 and 100,

  "reason": "Clear concise 1-2 sentence institutional rationale",

  "warnings": "Any key risk factors"

}}

"""



    def _call_gemini_api(self, prompt: str) -> AIDecision:

        """Query Gemini API with automatic API key rotation on 429 quota and model fallback."""

        from google.genai import types

        candidate_models = [self.model_name]

        for fallback in (

            "gemini-flash-lite-latest",

            "gemini-2.5-flash-lite",

            "gemini-2.5-flash",

            "gemini-2.0-flash",

            "gemini-3.1-flash-lite",

        ):

            if fallback not in candidate_models:

                candidate_models.append(fallback)



        response = None

        last_error = None

        num_keys = len(self.api_keys)



        # Iterate across available keys in rotation pool

        for attempt in range(num_keys):

            key_idx = (self._current_key_idx + attempt) % num_keys

            active_key = self.api_keys[key_idx]

            client = self._clients.get(active_key)

            if not client:

                continue



            for m in candidate_models:

                try:

                    response = client.models.generate_content(

                        model=m,

                        contents=prompt,

                        config=types.GenerateContentConfig(

                            response_mime_type="application/json",

                            temperature=0.1,  # Strict, low-variance reasoning

                        ),

                    )

                    if response and response.text:

                        # Key succeeded! Update active key index

                        self._current_key_idx = key_idx

                        break

                except Exception as e:

                    last_error = e

                    err_str = str(e).lower()

                    if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:

                        masked = f"{active_key[:6]}...{active_key[-4:]}"

                        logger.warning(

                            f"Gemini API key [{masked}] quota exhausted (429 RESOURCE_EXHAUSTED) on model '{m}'. "

                            f"Rotating to next available key in pool..."

                        )

                        break  # Break model loop to rotate key immediately

                    else:

                        logger.warning(f"Gemini model '{m}' evaluation failed on key [{active_key[:6]}...]: {e}. Trying next candidate model...")

                        continue



            if response and response.text:

                break



        if response is None or not response.text:

            raise last_error or RuntimeError("No response from Gemini models across all keys")



        data = json.loads(response.text)

        confirmed = bool(data.get("confirmed", False))

        confidence = float(data.get("confidence", 50.0))

        reason = str(data.get("reason", "AI evaluation complete."))

        warnings = str(data.get("warnings", ""))



        # Check against threshold

        if confidence < self.threshold:

            confirmed = False

            reason = f"AI confidence ({confidence:.1f}%) below threshold ({self.threshold}%): {reason}"



        return AIDecision(

            confirmed=confirmed,

            confidence=confidence,

            reason=reason,

            warnings=warnings,

        )



    def _heuristic_evaluation(

        self,

        signal: TradeSignal,

        htf_analysis: HTFAnalysis,

        current_spread: float,

    ) -> AIDecision:

        """

        High-conviction heuristic evaluation for when API keys are not connected.

        Scores structural alignment, liquidity sweep depth, and R:R asymmetry.

        """

        score = 0.0

        reasons = []



        # 1. Macro Trend Confluence (+35 pts)

        if signal.htf_bias == MarketBias.BULLISH and signal.direction == Direction.BUY:

            score += 35.0

            reasons.append("Macro bullish alignment above 200 EMA")

        elif signal.htf_bias == MarketBias.BEARISH and signal.direction == Direction.SELL:

            score += 35.0

            reasons.append("Macro bearish alignment below 200 EMA")

        else:

            return AIDecision(

                confirmed=False,

                confidence=30.0,

                reason="AI Rejected: Setup opposes macro institutional order flow.",

            )



        # 2. SMC / ICT Confirmation Type (+20-35 pts)

        conf_name = signal.ltf_confirmation.value

        if "OB_SCALP_5M" in conf_name:

            score += 35.0

            reasons.append("5M Break of Structure & Order Block retest scalp")

        elif "ICT_SILVER_BULLET" in conf_name:

            score += 35.0

            reasons.append("ICT Silver Bullet institutional liquidity sweep & FVG model")

        elif "ICT_KILLZONE_FVG" in conf_name:

            score += 35.0

            reasons.append("ICT Kill Zone high-volume Fair Value Gap retest")

        elif "ICT_JUDAS_SWING" in conf_name:

            score += 35.0

            reasons.append("ICT Judas Swing liquidity purge reversal")

        elif "ICT_OTE_RETEST" in conf_name:

            score += 30.0

            reasons.append("ICT Optimal Trade Entry (OTE) Fib confluence retest")

        elif "OB_PLUS_FVG" in conf_name:

            score += 35.0

            reasons.append("Order Block + Fair Value Gap confluence retest")

        elif "OF_LIQUIDITY_TRAP" in conf_name:

            score += 35.0

            reasons.append("Order Flow Liquidity Trap stop-run & delta reversal")

        elif "OF_ABSORPTION" in conf_name:

            score += 35.0

            reasons.append("Order Flow institutional passive limit absorption")

        elif "OF_DELTA_DIVERGENCE" in conf_name:

            score += 30.0

            reasons.append("Order Flow Cumulative Volume Delta (CVD) divergence")

        elif "OB_MITIGATION" in conf_name:

            score += 30.0

            reasons.append("Clean Order Block mitigation")

        elif "FVG_MITIGATION" in conf_name:

            score += 25.0

            reasons.append("Fair Value Gap fill")

        elif "LIQUIDITY_SWEEP" in conf_name:

            score += 20.0

            reasons.append("Liquidity sweep confirmed")

        elif "PULLBACK" in conf_name or "STRUCTURAL_BREAK" in conf_name:

            score += 20.0

            reasons.append("Structural trend continuation")



        # 3. Risk-Reward Asymmetry (+20 pts)

        if "OB_SCALP_5M" in conf_name:

            if signal.rr_ratio >= 1.5:

                score += 20.0

                reasons.append(f"Optimal 5M Scalp R:R ({signal.rr_ratio:.2f}:1)")

            elif signal.rr_ratio >= 1.4:

                score += 15.0

                reasons.append(f"Acceptable 5M Scalp R:R ({signal.rr_ratio:.2f}:1)")

        elif "ICT" in conf_name:

            if signal.rr_ratio >= 2.0:

                score += 20.0

                reasons.append(f"Optimal ICT Model R:R ({signal.rr_ratio:.2f}:1)")

            elif signal.rr_ratio >= 1.5:

                score += 15.0

                reasons.append(f"Acceptable ICT Model R:R ({signal.rr_ratio:.2f}:1)")

        elif "OF_" in conf_name or "ORDER_FLOW" in getattr(signal, 'strategy_id', ''):

            if signal.rr_ratio >= 2.0:

                score += 20.0

                reasons.append(f"Optimal Order Flow R:R ({signal.rr_ratio:.2f}:1)")

            elif signal.rr_ratio >= 1.5:

                score += 15.0

                reasons.append(f"Acceptable Order Flow R:R ({signal.rr_ratio:.2f}:1)")

        else:

            if signal.rr_ratio >= 3.0:

                score += 20.0

                reasons.append(f"Strong asymmetric R:R ({signal.rr_ratio:.2f}:1)")

            elif signal.rr_ratio >= 2.5:

                score += 15.0

                reasons.append(f"Acceptable R:R ({signal.rr_ratio:.2f}:1)")



        # 4. Invalidation quality check

        if signal.sl_distance > 0:

            score += 10.0



        confidence = min(98.0, round(score, 1))

        confirmed = confidence >= self.threshold



        summary_reason = "; ".join(reasons)

        if not confirmed:

            summary_reason = f"Confidence ({confidence}%) below threshold ({self.threshold}%). Insufficient confluence."



        return AIDecision(

            confirmed=confirmed,

            confidence=confidence,

            reason=summary_reason,

            warnings="Evaluated via internal heuristic AI analyzer.",

        )

