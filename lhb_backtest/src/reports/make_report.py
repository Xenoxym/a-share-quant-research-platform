"""HTML report generation for backtest results."""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from jinja2 import Template

from src.reports.plot_results import plot_return_distribution
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from src.backtest.filter_engine import BacktestResult

logger = get_logger("make_report")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
            line-height: 1.6;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .header h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
        .header .timestamp { opacity: 0.85; font-size: 0.9rem; }
        .card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
        }
        .card h2 {
            font-size: 1.3rem;
            color: #34495e;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
        }
        .filters-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 0.8rem;
        }
        .filter-item {
            background: #f8f9fa;
            padding: 0.6rem 1rem;
            border-radius: 6px;
            font-size: 0.9rem;
            border-left: 3px solid #667eea;
        }
        .filter-item .key { font-weight: 600; color: #555; }
        .filter-item .value { color: #667eea; font-weight: 500; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        th, td { padding: 0.7rem 1rem; text-align: left; }
        th {
            background: #f1f3f5;
            font-weight: 600;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
        }
        td { border-bottom: 1px solid #f1f3f5; }
        tr:hover td { background: #f8f9fa; }
        .stat-positive { color: #27ae60; font-weight: 500; }
        .stat-negative { color: #e74c3c; font-weight: 500; }
        .chart-container {
            text-align: center;
            margin: 1rem 0;
        }
        .chart-container img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .footer {
            text-align: center;
            color: #95a5a6;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #ecf0f1;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        .metric-card .label { font-size: 0.8rem; color: #7f8c8d; margin-bottom: 0.3rem; }
        .metric-card .value { font-size: 1.4rem; font-weight: 700; color: #2c3e50; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <div class="timestamp">Generated: {{ timestamp }}</div>
        </div>

        <div class="card">
            <p>本报告是事件价格标签统计，不是组合净值。未模拟跌停或停牌退出延期、资金占用及完整公司行动；减去假设成本不代表已实现成交收益。样本按已观察标签计算。</p>
        </div>

        <div class="card">
            <h2>Filter Parameters</h2>
            {% if filters %}
            <div class="filters-grid">
                {% for key, value in filters.items() %}
                <div class="filter-item">
                    <span class="key">{{ key }}:</span>
                    <span class="value">{{ value }}</span>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p>No filters applied (all events included)</p>
            {% endif %}
        </div>

        <div class="card">
            <h2>Key Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="label">Sample Count</div>
                    <div class="value">{{ sample_count }}</div>
                </div>
                {% for metric in key_metrics %}
                <div class="metric-card">
                    <div class="label">{{ metric.label }}</div>
                    <div class="value {% if metric.positive %}stat-positive{% elif metric.negative %}stat-negative{% endif %}">
                        {{ metric.value }}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="card">
            <h2>Summary Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in stats_rows %}
                    <tr>
                        <td>{{ row.metric }}</td>
                        <td class="{% if row.css_class %}{{ row.css_class }}{% endif %}">
                            {{ row.value }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        {% for chart in charts %}
        <div class="card">
            <h2>{{ chart.title }}</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{{ chart.data }}" alt="{{ chart.title }}">
            </div>
        </div>
        {% endfor %}

        <div class="footer">
            <p>事件价格标签研究，不是组合净值；尚未计入全部权益事件及成交约束。盘后事件收盘基准指标仅供研究。</p>
            <p>LHB Backtest Report | A-Share Dragon-Tiger Event Backtesting System</p>
        </div>
    </div>
</body>
</html>"""


def generate_html_report(
    backtest_result: "BacktestResult",
    output_dir: str | None = None,
    title: str = "LHB Backtest Report",
) -> str:
    """Generate an HTML report from a BacktestResult.

    Parameters
    ----------
    backtest_result : BacktestResult
        Backtest result container with summary and detail data.
    output_dir : str or None
        Directory to save report. Default: data/output/.
    title : str
        Report title.

    Returns
    -------
    str
        Path to the generated HTML report file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare output directory
    if output_dir is None:
        output_dir = "data/output"
    out_path = ensure_dir(output_dir)
    report_path = out_path / f"report_{file_timestamp}.html"

    # Prepare statistics rows
    stats_rows = _build_stats_rows(backtest_result.summary)

    # Prepare key metrics
    key_metrics = _build_key_metrics(backtest_result.summary)

    # Generate charts
    charts = []
    detail = backtest_result.detail

    if len(detail) > 0 and "future_return_1d" in detail.columns:
        try:
            fig = plot_return_distribution(
                detail,
                return_col="future_return_1d",
                title="Next-Day Return Distribution",
            )
            charts.append({"title": "Next-Day Return Distribution", "data": _fig_to_base64(fig)})
        except Exception as e:
            logger.warning(f"Failed to generate return distribution chart: {e}")

    if len(detail) > 0 and "future_return_5d" in detail.columns:
        try:
            fig = plot_return_distribution(
                detail,
                return_col="future_return_5d",
                title="5-Day Return Distribution",
            )
            charts.append({"title": "5-Day Return Distribution", "data": _fig_to_base64(fig)})
        except Exception as e:
            logger.warning(f"Failed to generate 5d return distribution chart: {e}")

    # Render HTML
    template = Template(_HTML_TEMPLATE, autoescape=True)
    html_content = template.render(
        title=title,
        timestamp=timestamp,
        filters=backtest_result.filters,
        sample_count=f"{backtest_result.sample_count:,}",
        key_metrics=key_metrics,
        stats_rows=stats_rows,
        charts=charts,
    )

    report_path.write_text(html_content, encoding="utf-8")
    logger.info(f"HTML report saved to {report_path}")
    return str(report_path)


def _fig_to_base64(fig: "plt.Figure") -> str:
    """Convert matplotlib figure to base64 string for HTML embedding."""
    import matplotlib.pyplot as plt

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _format_pct(val: float) -> str:
    """Format a float as percentage string: 0.1234 -> '12.34%'."""
    if pd.isna(val):
        return "N/A"
    return f"{val * 100:.2f}%"


def _format_number(val: float, decimals: int = 2) -> str:
    """Format number with thousand separators."""
    if pd.isna(val):
        return "N/A"
    if abs(val) >= 1000:
        return f"{val:,.{decimals}f}"
    return f"{val:.{decimals}f}"


def _build_stats_rows(summary: pd.DataFrame) -> list[dict[str, str]]:
    """Build formatted statistics rows for the HTML table."""
    if summary.empty:
        return []

    rows = []
    for col in summary.columns:
        val = summary[col].iloc[0]
        if col == "sample_count":
            continue

        metric_name = col.replace("_", " ").title()
        css_class = ""

        if "rate" in col or "win_rate" in col:
            formatted = _format_pct(val)
            if not pd.isna(val):
                css_class = "stat-positive" if val > 0.5 else ""
        elif "return" in col or "ratio" in col:
            formatted = _format_number(val, 4)
            if not pd.isna(val):
                css_class = "stat-positive" if val > 0 else "stat-negative"
        elif "drawdown" in col:
            formatted = _format_number(val, 4)
            css_class = "stat-negative" if not pd.isna(val) and val < 0 else ""
        else:
            formatted = _format_number(val, 4)

        rows.append({"metric": metric_name, "value": formatted, "css_class": css_class})

    return rows


def _build_key_metrics(summary: pd.DataFrame) -> list[dict[str, Any]]:
    """Build key metric cards for the report header."""
    if summary.empty:
        return []

    metrics = []
    highlight_cols = [
        ("next_day_limit_up_rate", "Next-Day Limit Up"),
        ("oo_win_rate_1d", "T+1 Open to T+2 Open Win Rate"),
        ("net_oo_return_1d", "Overnight Price Label, Less Assumed Cost"),
        ("entry_win_rate_5d", "T+1 Open to T+5 Close Win Rate"),
        ("net_entry_return_5d", "5D Entry Price Label, Less Assumed Cost"),
        ("profit_loss_ratio_1d", "P/L Ratio 1D"),
    ]

    for col, label in highlight_cols:
        if col not in summary.columns:
            continue
        val = summary[col].iloc[0]
        if pd.isna(val):
            continue

        if "rate" in col:
            formatted = _format_pct(val)
            positive = val > 0.5
            negative = val < 0.3
        elif "return" in col:
            formatted = _format_pct(val)
            positive = val > 0
            negative = val < 0
        else:
            formatted = _format_number(val, 2)
            positive = val > 1
            negative = val < 1

        metrics.append({
            "label": label,
            "value": formatted,
            "positive": positive,
            "negative": negative,
        })

    return metrics
