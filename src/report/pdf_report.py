"""PDF report generation (ReportLab), mirroring the "Отчёт" design layout:
branded header, slide + heatmap image, top attention zones, model verdict,
SPPR disclaimer, case metadata, and physician signature fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"

# ReportLab's built-in base-14 fonts ("Helvetica" etc.) only cover Latin
# WinAnsiEncoding — every Cyrillic character in this report (labels,
# disclaimer, tissue type...) rendered as a solid black glyph-missing box
# under them, discovered by actually opening a generated PDF, not just
# checking the HTTP status. DejaVu Sans has full Cyrillic coverage and is
# already bundled as TTF data with matplotlib (an existing transitive
# dependency via grad-cam) — reusing it needs no new dependency or binary
# asset committed to the repo, unlike shipping our own font file.
def _register_fonts() -> str:
    """Returns the font family name to actually use: "DejaVuSans" if
    registration succeeds, or ReportLab's built-in "Helvetica" (always
    available, but Latin-only — Cyrillic renders as missing-glyph boxes
    under it) if matplotlib's bundled TTFs can't be found for some reason.
    """
    family = "DejaVuSans"
    try:
        import matplotlib

        ttf_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        for name, filename in [
            (family, "DejaVuSans.ttf"),
            (f"{family}-Bold", "DejaVuSans-Bold.ttf"),
            (f"{family}-Oblique", "DejaVuSans-Oblique.ttf"),
            (f"{family}-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
        ]:
            pdfmetrics.registerFont(TTFont(name, str(ttf_dir / filename)))
        pdfmetrics.registerFontFamily(
            family,
            normal=family, bold=f"{family}-Bold",
            italic=f"{family}-Oblique", boldItalic=f"{family}-BoldOblique",
        )
        return family
    except Exception:
        logger.exception(
            "Could not register DejaVu Sans for Cyrillic PDF text — falling back to "
            "Helvetica, Russian text will render as missing-glyph boxes."
        )
        return "Helvetica"


_FONT = _register_fonts()
_FONT_BOLD = f"{_FONT}-Bold"
_FONT_OBLIQUE = f"{_FONT}-Oblique"

# Approximate hex equivalents of the frontend's OKLCH brand palette — ReportLab
# has no OKLCH support, so these are print-only approximations of the values
# used natively in frontend/src/styles/tokens.css.
_BRAND_PRIMARY = HexColor("#6D3FD9")
_MALIGNANT_COLOR = HexColor("#C23B22")
_BENIGN_COLOR = HexColor("#2F8F4E")
_MUTED_TEXT = HexColor("#6B7280")
_DARK_TEXT = HexColor("#1F2430")
_BORDER = HexColor("#DDE1E8")
_PANEL_BG = HexColor("#F7F7FA")


@dataclass(frozen=True)
class ReportRegion:
    rank: int
    x: int
    y: int
    score: float


@dataclass(frozen=True)
class ReportData:
    case_id: str
    created_at: str  # pre-formatted, e.g. "01.08.2026, 09:42"
    tissue_type: str
    source_filename: str
    analysis_mode: str  # "Живой анализ" | "Полный препарат"
    verdict_label: str  # e.g. "Злокачественная" / "Доброкачественная"
    is_malignant: bool
    confidence: float
    malignant_probability: float
    benign_probability: float
    top_regions: list[ReportRegion]
    heatmap_image_path: Path
    disclaimer: str
    model_version: str
    generated_at: str
    doctor_name: str
    logo_path: Path | None = None


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "doc_title": ParagraphStyle("doc_title", fontName=_FONT_BOLD, fontSize=13,
                                     textColor=_DARK_TEXT, alignment=2, leading=15),
        "doc_meta": ParagraphStyle("doc_meta", fontName=_FONT, fontSize=8.5,
                                    textColor=_MUTED_TEXT, alignment=2, leading=11),
        "label": ParagraphStyle("label", fontName=_FONT_BOLD, fontSize=8,
                                 textColor=_MUTED_TEXT, leading=10, spaceAfter=4),
        "verdict": ParagraphStyle("verdict", fontName=_FONT_BOLD, fontSize=17,
                                   leading=20),
        "verdict_sub": ParagraphStyle("verdict_sub", fontName=_FONT, fontSize=9,
                                       textColor=_MUTED_TEXT, leading=12),
        "body": ParagraphStyle("body", fontName=_FONT, fontSize=9,
                                textColor=_DARK_TEXT, leading=13),
        "disclaimer": ParagraphStyle("disclaimer", fontName=_FONT_OBLIQUE, fontSize=8,
                                      textColor=_MUTED_TEXT, leading=12),
        "small": ParagraphStyle("small", fontName=_FONT, fontSize=8,
                                 textColor=_MUTED_TEXT, leading=11),
        "table_cell": ParagraphStyle("table_cell", fontName=_FONT, fontSize=9,
                                      textColor=_DARK_TEXT),
    }


def _is_readable_image(path: Path) -> bool:
    """`.exists()` alone isn't enough — a present-but-corrupt/truncated file
    (seen in practice: src/report/assets/logo.png) makes ReportLab's
    drawImage() raise deep inside rendering, well past the point where a
    fallback could still be swapped in. Decode it up front instead."""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def _build_header(data: ReportData, styles: dict) -> Table:
    logo_path = data.logo_path or _DEFAULT_LOGO_PATH
    logo_cell = (
        RLImage(str(logo_path), width=1.3 * inch, height=0.26 * inch)
        if logo_path and logo_path.exists() and _is_readable_image(logo_path)
        else Paragraph("HistoVision", ParagraphStyle("logo_fallback", fontName=_FONT_BOLD,
                                                       fontSize=14, textColor=_BRAND_PRIMARY))
    )
    meta = [
        Paragraph("Заключение по анализу препарата", styles["doc_title"]),
        Paragraph(f"Случай {data.case_id} · {data.created_at}", styles["doc_meta"]),
    ]
    header = Table([[logo_cell, meta]], colWidths=[3.2 * inch, 4.0 * inch])
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, -1), 1.2, _DARK_TEXT),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return header


def _build_left_column(data: ReportData, styles: dict) -> list:
    elements = [Paragraph("ИЗОБРАЖЕНИЕ ПРЕПАРАТА С ТЕПЛОВОЙ КАРТОЙ", styles["label"])]
    if data.heatmap_image_path.exists() and _is_readable_image(data.heatmap_image_path):
        elements.append(RLImage(str(data.heatmap_image_path), width=3.35 * inch, height=2.7 * inch))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("ТОП-3 ЗОНЫ ВНИМАНИЯ", styles["label"]))

    rows = [["#", "Координаты", "Значимость"]]
    for region in data.top_regions:
        rows.append([str(region.rank), f"X: {region.x}, Y: {region.y}", f"{region.score:.1%}"])
    zones_table = Table(rows, colWidths=[0.3 * inch, 1.9 * inch, 1.15 * inch])
    zones_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), _FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), _MUTED_TEXT),
        ("TEXTCOLOR", (-1, 1), (-1, -1), _MALIGNANT_COLOR),
        ("FONTNAME", (-1, 1), (-1, -1), _FONT_BOLD),
        ("ALIGN", (-1, 0), (-1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, _BORDER),
        ("LINEBELOW", (0, 1), (-1, -2), 0.5, _BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(zones_table)
    return elements


def _build_verdict_panel(data: ReportData, styles: dict) -> Table:
    verdict_color = _MALIGNANT_COLOR if data.is_malignant else _BENIGN_COLOR
    verdict_style = ParagraphStyle("verdict_colored", parent=styles["verdict"], textColor=verdict_color)

    content = [
        Paragraph("ВЕРДИКТ МОДЕЛИ", styles["label"]),
        Paragraph(data.verdict_label, verdict_style),
        Paragraph(f"уверенность модели <b>{data.confidence:.1%}</b>", styles["verdict_sub"]),
        Spacer(1, 8),
        _probability_bar("Злокачественная", data.malignant_probability, _MALIGNANT_COLOR, styles),
        Spacer(1, 4),
        _probability_bar("Доброкачественная", data.benign_probability, _BENIGN_COLOR, styles),
    ]
    panel = Table([[content]], colWidths=[3.35 * inch])
    panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _PANEL_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, _BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return panel


def _probability_bar(label: str, value: float, color: HexColor, styles: dict) -> Table:
    label_row = Table(
        [[Paragraph(label, styles["small"]), Paragraph(f"<b>{value:.1%}</b>", styles["small"])]],
        colWidths=[2.3 * inch, 1.0 * inch],
    )
    label_row.setStyle(TableStyle([
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))

    bar_width = 3.3 * inch
    filled = max(bar_width * min(max(value, 0.0), 1.0), 2)
    bar = Table([[""]], colWidths=[filled], rowHeights=[5])
    bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    track = Table([[bar]], colWidths=[bar_width], rowHeights=[5])
    track.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E7E8EE")),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    wrapper = Table([[label_row], [track]], colWidths=[3.3 * inch])
    wrapper.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return wrapper


def _build_right_column(data: ReportData, styles: dict) -> list:
    elements = [_build_verdict_panel(data, styles), Spacer(1, 10)]
    elements.append(Paragraph(data.disclaimer, styles["disclaimer"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("СВЕДЕНИЯ О СЛУЧАЕ", styles["label"]))

    info_rows = [
        ["Ткань", data.tissue_type],
        ["Файл препарата", data.source_filename],
        ["Режим анализа", data.analysis_mode],
    ]
    info_table = Table(
        [[Paragraph(k, styles["small"]), Paragraph(f"<b>{v}</b>", styles["small"])] for k, v in info_rows],
        colWidths=[1.5 * inch, 1.85 * inch],
    )
    info_table.setStyle(TableStyle([
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    elements.append(info_table)
    return elements


def _build_body(data: ReportData, styles: dict) -> Table:
    left = _build_left_column(data, styles)
    right = _build_right_column(data, styles)
    body = Table([[left, right]], colWidths=[3.5 * inch, 3.5 * inch])
    body.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 22),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return body


def _build_footer(data: ReportData, styles: dict) -> Table:
    meta = Paragraph(f"HistoVision · сформировано автоматически {data.generated_at}", styles["small"])

    signature_line_style = TableStyle([
        ("LINEBELOW", (0, 0), (0, 0), 0.75, _MUTED_TEXT),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])
    doctor_sig = Table([[""]], colWidths=[2.3 * inch], rowHeights=[22])
    doctor_sig.setStyle(signature_line_style)
    date_sig = Table([[""]], colWidths=[1.4 * inch], rowHeights=[22])
    date_sig.setStyle(signature_line_style)

    signatures = Table(
        [[doctor_sig, date_sig], [Paragraph("подпись врача", styles["small"]),
                                   Paragraph("дата", styles["small"])]],
        colWidths=[2.3 * inch, 1.4 * inch],
    )
    signatures.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    footer = Table([[meta, signatures]], colWidths=[3.5 * inch, 3.7 * inch])
    footer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.75, _BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return footer


def generate_pdf_report(data: ReportData, output_path: Path) -> Path:
    """Render the one-page case report to `output_path` and return it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()

    doc = SimpleDocTemplate(
        str(output_path), pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.55 * inch, bottomMargin=0.55 * inch,
        title=f"HistoVision — {data.case_id}",
    )

    story = [
        _build_header(data, styles),
        Spacer(1, 14),
        _build_body(data, styles),
        Spacer(1, 16),
        _build_footer(data, styles),
    ]
    doc.build(story)
    logger.info("Generated PDF report for case %s at %s", data.case_id, output_path)
    return output_path
