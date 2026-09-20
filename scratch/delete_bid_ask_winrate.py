import re

path = 'templates/index.html'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Bento KPI Card container from grid-cols-1 md:grid-cols-3 gap-4 to grid-cols-1 md:grid-cols-2 gap-5
content = content.replace(
    '<div class="grid grid-cols-1 md:grid-cols-3 gap-4">\n          <!-- Card 1: Total Equity & Balance -->',
    '<div class="grid grid-cols-1 md:grid-cols-2 gap-5">\n          <!-- Card 1: Total Equity & Balance -->'
)

# 2. Delete Bento Card 2: Today's Win Rate with Radial Gauge
card2_pattern = re.compile(
    r'\s*<!-- Card 2: Win Rate with Radial Gauge -->\s*<div class="bg-white rounded-2xl p-5 shadow-\[0_4px_20px_-2px_rgba\(124,58,237,0\.06\)\] border border-purple-100.*?Avg Loss:</span>\s*<span id="kpiAvgLoss".*?</div>\s*</div>\s*</div>',
    re.DOTALL
)
match = card2_pattern.search(content)
if match:
    content = content[:match.start()] + content[match.end():]
    print("[1/6] Card 2 (Today's Win Rate) removed.")
else:
    print("[WARN] Card 2 pattern did not match!")

# 3. Delete Strategy Win Rate rows in the 4 execution engines
strat_wr_pattern = re.compile(
    r'\s*<div class="flex justify-between">\s*<span class="text-slate-800 font-extrabold">Win Rate:</span>\s*<span id="(?:smc|scalp|ict|of)WinRate".*?</div>',
    re.DOTALL
)
found_strat_wr = len(strat_wr_pattern.findall(content))
content = strat_wr_pattern.sub('', content)
print(f"[2/6] Removed {found_strat_wr} strategy Win Rate rows (expected 4).")

# 4. In Strategy 4 footer, change BID HEAVY to BUY HEAVY
content = content.replace('BID HEAVY', 'BUY HEAVY')
print("[3/6] Replaced BID HEAVY with BUY HEAVY.")

# 5. In the 5 Multi-Pair Scanner cards, remove the BID / ASK divs
bid_ask_pattern = re.compile(
    r'\s*<div class="flex justify-between text-slate-600 font-bold">\s*<span>BID / ASK:</span>\s*<span class="text-slate-950 font-black">.*?</span>\s*</div>',
    re.DOTALL
)
found_bid_ask = len(bid_ask_pattern.findall(content))
content = bid_ask_pattern.sub('', content)
print(f"[4/6] Removed {found_bid_ask} BID / ASK rows in pair cards (expected 5).")

# 6. Tab 2 (Performance Analytics):
# 6a. Delete Card 3 (Win Rate & Differential)
card3_pattern = re.compile(
    r'\s*<!-- 3\. Win Rate & Differential -->\s*<div class="bg-white border border-purple-100 rounded-2xl p-5 shadow-sm">.*?<span>Net Balance \(Wins [^)]+\):</span>\s*<span id="analyticsNetSpread".*?</div>\s*</div>',
    re.DOTALL
)
match_card3 = card3_pattern.search(content)
if match_card3:
    content = content[:match_card3.start()] + content[match_card3.end():]
    print("[5a/6] Analytics Card 3 (Win Rate & Differential) removed.")
else:
    print("[WARN] Analytics Card 3 pattern did not match!")

# 6b. Update Tab 2 Primary Metric Cards grid from 6 cards to 5 cards (grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4)
content = content.replace(
    '<!-- 6 Primary Metric Cards -->\n        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">',
    '<!-- Primary Metric Cards -->\n        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">'
)

# 6c. Remove Win Rate column from Strategy Table and Pair Table in Tab 2
content = content.replace(
    '<th class="py-3 px-3 text-center">Win Rate</th>',
    ''
)
content = content.replace(
    '<tr><td colspan="5" class="py-5 text-center text-slate-500 font-sans font-semibold">No strategy trades recorded.</td></tr>',
    '<tr><td colspan="4" class="py-5 text-center text-slate-500 font-sans font-semibold">No strategy trades recorded.</td></tr>'
)
content = content.replace(
    '<tr><td colspan="5" class="py-5 text-center text-slate-500 font-sans font-semibold">No pair trades recorded.</td></tr>',
    '<tr><td colspan="4" class="py-5 text-center text-slate-500 font-sans font-semibold">No pair trades recorded.</td></tr>'
)

# 6d. Tab 3 Trade History Header: remove Win Rate: ---%
content = content.replace(
    '<div class="font-body-sm text-sm sm:text-base text-slate-800 font-black">\n                Win Rate: <span id="historyWinRate" class="font-black text-emerald-700 font-mono">---%</span> \n                <span id="historyWinLoss" class="text-xs sm:text-sm text-slate-600 font-extrabold">(0W / 0L)</span>\n              </div>',
    '<div class="font-body-sm text-sm sm:text-base text-slate-800 font-black">\n                Closed Trades: <span id="historyWinLoss" class="text-slate-950 font-mono font-black">(0W / 0L)</span>\n              </div>'
)

# 6e. Clean up JS in fetchAnalytics for table rows
js_strat_row_old = '<td class="py-3 px-3 text-center font-mono ${(s.win_rate || 0) >= 50 ? \'text-emerald-700\' : \'text-rose-700\'} font-black">${(s.win_rate || 0).toFixed(1)}%</td>'
content = content.replace(js_strat_row_old, '')

js_pair_row_old = '<td class="py-3 px-3 text-center font-mono ${(p.win_rate || 0) >= 50 ? \'text-emerald-700\' : \'text-rose-700\'} font-black">${(p.win_rate || 0).toFixed(1)}%</td>'
content = content.replace(js_pair_row_old, '')

# Guard winRateEl and diff elements in fetchAnalytics
old_js_analytics_wr = """        // 3. Win Rate & Differential
        const winRateEl = document.getElementById('analyticsWinRate');
        const diffBadge = document.getElementById('analyticsDiffBadge');
        const netSpread = document.getElementById('analyticsNetSpread');
        const winLossSpread = document.getElementById('analyticsWinLossSpread');

        const winRate = data.win_rate || 0.0;
        winRateEl.innerText = `${winRate.toFixed(1)}%`;
        winRateEl.className = `text-3xl sm:text-4xl font-black font-mono ${winRate >= 50 ? 'text-emerald-700' : (data.total_closed_trades > 0 ? 'text-rose-700' : 'text-slate-950')}`;

        winLossSpread.innerText = `(${data.total_tp || 0}W / ${data.total_sl || 0}L)`;

        const diff = data.win_loss_diff !== undefined ? data.win_loss_diff : ((data.total_tp || 0) - (data.total_sl || 0));
        if (diff > 0) {
          diffBadge.innerText = `+${diff} Net Wins`;
          diffBadge.className = "px-3 py-1 rounded text-xs sm:text-sm font-black bg-emerald-50 border border-emerald-300 text-emerald-900";
          netSpread.innerText = `+${diff} (Positive Edge)`;
          netSpread.className = "font-black text-emerald-700 font-mono";
        } else if (diff < 0) {
          diffBadge.innerText = `${diff} Net Losses`;
          diffBadge.className = "px-3 py-1 rounded text-xs sm:text-sm font-black bg-rose-50 border border-rose-300 text-rose-900";
          netSpread.innerText = `${diff} (Negative Drawdown)`;
          netSpread.className = "font-black text-rose-700 font-mono";
        } else {
          diffBadge.innerText = "0 Neutral";
          diffBadge.className = "px-3 py-1 rounded text-xs sm:text-sm font-black bg-purple-50 text-slate-700 border border-purple-200";
          netSpread.innerText = "0 (Even Break)";
          netSpread.className = "font-black text-slate-800 font-mono";
        }"""

new_js_analytics_wr = """        // 3. Trade Edge & Differential (null guarded)
        const winRateEl = document.getElementById('analyticsWinRate');
        const diffBadge = document.getElementById('analyticsDiffBadge');
        const netSpread = document.getElementById('analyticsNetSpread');
        const winLossSpread = document.getElementById('analyticsWinLossSpread');

        if (winRateEl) {
          const winRate = data.win_rate || 0.0;
          winRateEl.innerText = `${winRate.toFixed(1)}%`;
          winRateEl.className = `text-3xl sm:text-4xl font-black font-mono ${winRate >= 50 ? 'text-emerald-700' : (data.total_closed_trades > 0 ? 'text-rose-700' : 'text-slate-950')}`;
        }
        if (winLossSpread) winLossSpread.innerText = `(${data.total_tp || 0}W / ${data.total_sl || 0}L)`;

        if (diffBadge && netSpread) {
          const diff = data.win_loss_diff !== undefined ? data.win_loss_diff : ((data.total_tp || 0) - (data.total_sl || 0));
          if (diff > 0) {
            diffBadge.innerText = `+${diff} Net Wins`;
            diffBadge.className = "px-3 py-1 rounded text-xs sm:text-sm font-black bg-emerald-50 border border-emerald-300 text-emerald-900";
            netSpread.innerText = `+${diff} (Positive Edge)`;
            netSpread.className = "font-black text-emerald-700 font-mono";
          } else if (diff < 0) {
            diffBadge.innerText = `${diff} Net Losses`;
            diffBadge.className = "px-3 py-1 rounded text-xs sm:text-sm font-black bg-rose-50 border border-rose-300 text-rose-900";
            netSpread.innerText = `${diff} (Negative Drawdown)`;
            netSpread.className = "font-black text-rose-700 font-mono";
          } else {
            diffBadge.innerText = "0 Neutral";
            diffBadge.className = "px-3 py-1 rounded text-xs sm:text-sm font-black bg-purple-50 text-slate-700 border border-purple-200";
            netSpread.innerText = "0 (Even Break)";
            netSpread.className = "font-black text-slate-800 font-mono";
          }
        }"""

if old_js_analytics_wr in content:
    content = content.replace(old_js_analytics_wr, new_js_analytics_wr)
    print("[6/6] Guarded fetchAnalytics JS.")
else:
    print("[WARN] fetchAnalytics block did not match directly.")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Saved changes to templates/index.html")
