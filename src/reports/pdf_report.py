"""
reports/pdf_report.py
---------------------
Generates a professional multi-section PDF derivatives risk report using
the fpdf2 library.

Sections
--------
1. Header + report date
2. Executive summary (total MtM and daily P&L)
3. Greeks table (Current | Prior | Limit | Status)
4. Top-5 positions by absolute dollar vega
5. Market context (per-ticker daily moves)
6. AI-generated Claude commentary (4 paragraphs)
"""

import logging
import os
from typing import Optional

from fpdf import FPDF, XPos, YPos

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_DARK_BLUE = (15, 40, 80)
_MID_BLUE = (30, 80, 160)
_LIGHT_BLUE = (210, 225, 245)
_WHITE = (255, 255, 255)
_BLACK = (20, 20, 20)
_LIGHT_GREY = (245, 245, 245)
_MID_GREY = (180, 180, 180)
_RED = (200, 30, 30)
_GREEN = (20, 140, 60)
_AMBER = (200, 130, 0)

# ---------------------------------------------------------------------------
# Helper: format dollar amounts
# ---------------------------------------------------------------------------

def _fmt_dollar(val: float, decimals: int = 0) -> str:
    """Return a formatted dollar string, e.g. '$1,234,567'."""
    sign = "-" if val < 0 else ""
    return f"{sign}${abs(val):,.{decimals}f}"


def _fmt_pct(val: float) -> str:
    """Return a formatted percentage string, e.g. '-0.31%'."""
    return f"{val * 100:.2f}%"


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------

class DerivativesRiskReport(FPDF):
    """
    Custom FPDF subclass that provides styled header, footer, and section
    helpers used throughout the derivatives risk report.
    """

    def __init__(self, report_date: str):
        """
        Initialise the PDF with A4 portrait layout and store the report date.

        Parameters
        ----------
        report_date : str
            ISO-8601 date string embedded in the header and footer.
        """
        super().__init__(orientation="P", unit="mm", format="A4")
        self.report_date = report_date
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(left=15, top=15, right=15)

    def header(self) -> None:
        """Render the page header with firm branding and report title."""
        self.set_fill_color(*_DARK_BLUE)
        self.rect(0, 0, 210, 22, style="F")

        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*_WHITE)
        self.set_xy(15, 5)
        self.cell(130, 12, "DERIVATIVES RISK MANAGEMENT", align="L",
                  new_x=XPos.RIGHT, new_y=YPos.TOP)

        self.set_font("Helvetica", "", 9)
        self.set_xy(145, 5)
        self.cell(50, 6, f"Daily Risk Report", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_xy(145, 11)
        self.cell(50, 6, self.report_date, align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_text_color(*_BLACK)
        self.ln(6)

    def footer(self) -> None:
        """Render the page footer with page number and confidentiality note."""
        self.set_y(-12)
        self.set_fill_color(*_DARK_BLUE)
        self.rect(0, self.get_y(), 210, 12, style="F")
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_MID_GREY)
        self.cell(95, 8, "CONFIDENTIAL - FOR INTERNAL USE ONLY", align="L")
        self.cell(95, 8, f"Page {self.page_no()}", align="R")
        self.set_text_color(*_BLACK)

    # -----------------------------------------------------------------------
    # Section helpers
    # -----------------------------------------------------------------------

    def section_title(self, title: str) -> None:
        """
        Render a styled section heading.

        Parameters
        ----------
        title : str
            Section heading text.
        """
        self.ln(4)
        self.set_fill_color(*_MID_BLUE)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, f"  {title}", align="L", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*_BLACK)
        self.ln(2)

    def kv_row(self, label: str, value: str, value_color: Optional[tuple] = None) -> None:
        """
        Render a key-value row with optional colour on the value cell.

        Parameters
        ----------
        label : str
            Left-column label.
        value : str
            Right-column value text.
        value_color : tuple or None
            RGB colour tuple for the value cell.  Uses black by default.
        """
        self.set_font("Helvetica", "", 9)
        self.set_fill_color(*_LIGHT_GREY)
        self.cell(70, 7, label, align="L", fill=True,
                  new_x=XPos.RIGHT, new_y=YPos.TOP)
        if value_color:
            self.set_text_color(*value_color)
        self.cell(110, 7, value, align="L",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*_BLACK)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_pdf_report(
    payload: dict,
    commentary: str,
    output_path: str,
) -> None:
    """
    Build and save the complete derivatives risk PDF report.

    Parameters
    ----------
    payload : dict
        The structured report payload from
        :func:`engine.portfolio.build_report_payload`.
    commentary : str
        AI-generated risk narrative from
        :func:`narrator.claude_narrator.generate_risk_commentary`.
    output_path : str
        Full file-system path for the output PDF file.

    Returns
    -------
    None
        Writes the PDF to *output_path*.

    Raises
    ------
    OSError
        If the output directory cannot be created or the file cannot be
        written.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report_date = payload.get("report_date", "N/A")
    pm = payload.get("portfolio_metrics", {})
    current = pm.get("current", {})
    prior = pm.get("prior", {})
    attribution = pm.get("PnL_attribution", {})
    positions = payload.get("positions", [])
    market_ctx = payload.get("market_context", {})
    alerts = payload.get("alerts", [])

    from config import DELTA_LIMIT, VEGA_LIMIT

    pdf = DerivativesRiskReport(report_date)
    pdf.add_page()

    # -----------------------------------------------------------------------
    # Section 1: Executive Summary
    # -----------------------------------------------------------------------
    pdf.section_title("EXECUTIVE SUMMARY")

    total_mtm = current.get("total_MtM", 0.0)
    daily_pnl = current.get("daily_PnL", 0.0)
    pnl_color = _GREEN if daily_pnl >= 0 else _RED

    pdf.kv_row("Total Portfolio MtM:", _fmt_dollar(total_mtm))
    pdf.kv_row("Daily P&L:", _fmt_dollar(daily_pnl), value_color=pnl_color)
    pdf.kv_row(
        "P&L Attribution - Delta:",
        _fmt_dollar(attribution.get("delta_pnl", 0.0)),
    )
    pdf.kv_row(
        "P&L Attribution - Vega:",
        _fmt_dollar(attribution.get("vega_pnl", 0.0)),
    )
    pdf.kv_row(
        "P&L Attribution - Theta:",
        _fmt_dollar(attribution.get("theta_pnl", 0.0)),
    )
    pdf.kv_row("Active Positions:", str(len(positions)))
    pdf.kv_row("Alerts:", str(len(alerts)) + (" - see below" if alerts else " - none"))

    # -----------------------------------------------------------------------
    # Section 2: Greeks Table
    # -----------------------------------------------------------------------
    pdf.section_title("PORTFOLIO GREEKS")

    col_widths = [45, 38, 38, 38, 21]
    headers = ["Greek", "Current", "Prior Day", "Limit", "Status"]

    # Table header
    pdf.set_fill_color(*_LIGHT_BLUE)
    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, align="C", fill=True,
                 border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(8)

    greek_rows = [
        (
            "Net Dollar Delta",
            current.get("net_dollar_delta", 0.0),
            prior.get("net_dollar_delta", 0.0),
            DELTA_LIMIT,
        ),
        (
            "Net Dollar Vega",
            current.get("net_dollar_vega", 0.0),
            prior.get("net_dollar_vega", 0.0),
            VEGA_LIMIT,
        ),
        (
            "Net Dollar Gamma",
            current.get("net_dollar_gamma", 0.0),
            prior.get("net_dollar_gamma", 0.0),
            None,
        ),
        (
            "Net Dollar Theta",
            current.get("net_dollar_theta", 0.0),
            prior.get("net_dollar_theta", 0.0),
            None,
        ),
    ]

    pdf.set_font("Helvetica", "", 9)
    for idx, (name, curr_val, prior_val, limit) in enumerate(greek_rows):
        fill = _LIGHT_GREY if idx % 2 == 0 else _WHITE
        pdf.set_fill_color(*fill)

        # Determine breach status
        if limit is not None:
            breached = abs(curr_val) > limit
            status_text = "BREACH" if breached else "OK"
            status_color = _RED if breached else _GREEN
            limit_str = _fmt_dollar(limit)
        else:
            status_text = "N/A"
            status_color = _MID_GREY
            limit_str = "-"

        # Greek name cell
        pdf.cell(col_widths[0], 7, name, align="L", fill=True, border=1,
                 new_x=XPos.RIGHT, new_y=YPos.TOP)

        # Current
        pdf.set_text_color(*(_RED if curr_val < 0 else _BLACK))
        pdf.cell(col_widths[1], 7, _fmt_dollar(curr_val), align="R", fill=True, border=1,
                 new_x=XPos.RIGHT, new_y=YPos.TOP)

        # Prior
        pdf.set_text_color(*(_RED if prior_val < 0 else _BLACK))
        pdf.cell(col_widths[2], 7, _fmt_dollar(prior_val), align="R", fill=True, border=1,
                 new_x=XPos.RIGHT, new_y=YPos.TOP)

        # Limit
        pdf.set_text_color(*_BLACK)
        pdf.cell(col_widths[3], 7, limit_str, align="R", fill=True, border=1,
                 new_x=XPos.RIGHT, new_y=YPos.TOP)

        # Status
        pdf.set_text_color(*status_color)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(col_widths[4], 7, status_text, align="C", fill=True, border=1,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_BLACK)

    pdf.ln(3)

    # -----------------------------------------------------------------------
    # Section 3: Top-5 Positions by Absolute Dollar Vega
    # -----------------------------------------------------------------------
    pdf.section_title("TOP 5 POSITIONS BY DOLLAR VEGA")

    sorted_positions = sorted(
        positions, key=lambda p: abs(p.get("dollar_vega", 0.0)), reverse=True
    )[:5]

    pos_headers = ["Ticker", "Type", "Strike", "Expiry", "Qty", "MtM ($)", "$ Vega", "IV"]
    pos_widths = [20, 14, 20, 24, 14, 26, 26, 16]

    pdf.set_fill_color(*_LIGHT_BLUE)
    pdf.set_font("Helvetica", "B", 8)
    for i, h in enumerate(pos_headers):
        pdf.cell(pos_widths[i], 7, h, align="C", fill=True, border=1,
                 new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(7)

    pdf.set_font("Helvetica", "", 8)
    for idx, pos in enumerate(sorted_positions):
        fill = _LIGHT_GREY if idx % 2 == 0 else _WHITE
        pdf.set_fill_color(*fill)
        row_vals = [
            pos.get("ticker", ""),
            pos.get("option_type", "").upper(),
            f"${pos.get('strike', 0):.0f}",
            str(pos.get("expiry", "")),
            str(pos.get("quantity", 0)),
            _fmt_dollar(pos.get("MtM", 0.0)),
            _fmt_dollar(pos.get("dollar_vega", 0.0)),
            f"{pos.get('sigma', 0.0)*100:.1f}%",
        ]
        for i, val in enumerate(row_vals):
            pdf.cell(pos_widths[i], 7, val, align="C", fill=True, border=1,
                     new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(7)

    pdf.ln(3)

    # -----------------------------------------------------------------------
    # Section 4: Market Context
    # -----------------------------------------------------------------------
    pdf.section_title("MARKET CONTEXT - DAILY MOVES")

    pdf.set_font("Helvetica", "", 9)
    for i, (ticker, pct) in enumerate(market_ctx.items()):
        fill = _LIGHT_GREY if i % 2 == 0 else _WHITE
        pdf.set_fill_color(*fill)
        color = _GREEN if pct >= 0 else _RED
        pdf.cell(50, 7, ticker, align="L", fill=True, border=1,
                 new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_text_color(*color)
        pdf.cell(130, 7, _fmt_pct(pct), align="L", fill=True, border=1,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*_BLACK)

    pdf.ln(3)

    # -----------------------------------------------------------------------
    # Section 5: Alerts
    # -----------------------------------------------------------------------
    if alerts:
        pdf.section_title("RISK ALERTS")
        pdf.set_font("Helvetica", "", 9)
        for alert in alerts:
            pdf.set_fill_color(*_LIGHT_GREY)
            pdf.set_text_color(*_RED)
            pdf.cell(5, 7, "", new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_text_color(*_BLACK)
            pdf.multi_cell(175, 7, f"  [!] {alert}", align="L",
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    # -----------------------------------------------------------------------
    # Section 6: AI Risk Commentary
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.section_title("AI RISK COMMENTARY - CLAUDE ANALYSIS")

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*_MID_BLUE)
    pdf.cell(0, 6, "Generated by Claude (claude-sonnet-4-5) - Anthropic",
             align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*_BLACK)
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 9)
    # Split commentary into paragraphs and render each with spacing
    paragraphs = [p.strip() for p in commentary.split("\n\n") if p.strip()]
    for para in paragraphs:
        pdf.multi_cell(0, 5.5, para, align="J",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    pdf.output(output_path)
    logger.info("PDF report saved to: %s", output_path)
