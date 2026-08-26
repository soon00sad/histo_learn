"""PDF report generation (ReportLab), mirroring the "Отчёт" design layout:
branded header, slide with the segmentation mask overlaid, tissue-class
area breakdown, model verdict, SPPR disclaimer, case metadata, and
physician signature fields.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, UnidentifiedImageError
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

# Printed class breakdown is capped — a real result can carry all 16 BCSS
# classes (config/bcss_classes.yaml), most with a near-zero share; the web
# UI (SegmentationViewer.tsx) has room to list everything, a one-page PDF
# doesn't. The remainder is folded into a single summary line instead of
# silently dropped.
_MAX_CLASSES_SHOWN = 8


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
class ReportClassArea:
    name_ru: str
    color: str  # "#RRGGBB", from config/bcss_classes.yaml via src/utils/bcss_classes.py
    fraction: float


@dataclass(frozen=True)
class ReportData:
    case_id: str
    created_at: str  # pre-formatted, e.g. "01.08.2026, 09:42"
    tissue_type: str
    source_filename: str
    analysis_mode: str  # "Живой анализ" | "Полный препарат"
    verdict_label: str  # e.g. "Злокачественная" / "Доброкачественная"
    is_malignant: bool
    tumor_area_fraction: float
    class_areas: list[ReportClassArea]  # sorted descending by fraction
    source_image_path: Path
    mask_image_path: Path
    disclaimer: str
    model_version: str
    generated_at: str
    doctor_name: str
    logo_path: Path | None = None
    # "model" (real segmentation output) vs "bcss_ground_truth" (pathologist-
    # annotated reference mask — exhibition demo cases while the model is
    # still undertrained, see docs/MODEL.md). Printed prominently so a demo
    # report is never mistaken for a real model result.
    mask_source: str = "model"


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
        "reference_badge": ParagraphStyle("reference_badge", fontName=_FONT_BOLD, fontSize=8.5,
                                           textColor=_BRAND_PRIMARY, alignment=2, leading=11),
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


def _composite_mask_overlay(source_path: Path, mask_path: Path, opacity: float = 0.55) -> Image.Image | None:
    """Blend the segmentation mask over the source tissue image — mirrors
    the frontend's mix-blend-mode:multiply overlay
    (SegmentationViewer.tsx), so the printed report shows the same "mask
    on tissue" view instead of the mask in isolation. The mask is resized
    to the source's pixel size first: for a WSI case the two are saved at
    different resolutions (thumbnail vs. mask_output_downsample — see
    src/inference/wsi_segmenter.py) even though they cover the same
    extent, and for a patch case the mask is fixed at the model's
    input_size while the source keeps its original upload resolution.

    Returns None only if the source itself is missing/unreadable (nothing
    to show); if the mask is missing/unreadable, falls back to the plain
    source image rather than failing the whole report.
    """
    if not (source_path.exists() and _is_readable_image(source_path)):
        return None
    source = Image.open(source_path).convert("RGB")

    if not (mask_path.exists() and _is_readable_image(mask_path)):
        return source

    mask = Image.open(mask_path).convert("RGBA").resize(source.size, Image.NEAREST)
    mask_rgb = mask.convert("RGB")
    # UNCOVERED pixels (wsi_segmenter.UNCOVERED) are fully transparent in the
    # mask's own palette (see case_service.save_mask_png's tRNS chunk) — respect
    # that per-pixel alpha, on top of an overall opacity so tissue detail stays
    # visible under classified regions too.
    alpha = mask.split()[3].point(lambda a: int(a * opacity))
    multiplied = ImageChops.multiply(source, mask_rgb)
    return Image.composite(multiplied, source, alpha)


def _rl_image_from_pil(image: Image.Image, width: float, height: float) -> RLImage:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return RLImage(buffer, width=width, height=height)


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
    if data.mask_source != "model":
        meta.append(Paragraph("ПРИМЕР НА ЭТАЛОННЫХ ДАННЫХ BCSS — НЕ ВЫВОД МОДЕЛИ", styles["reference_badge"]))
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
    elements = [Paragraph("ИЗОБРАЖЕНИЕ ПРЕПАРАТА С МАСКОЙ СЕГМЕНТАЦИИ", styles["label"])]
    overlay = _composite_mask_overlay(data.source_image_path, data.mask_image_path)
    if overlay is not None:
        elements.append(_rl_image_from_pil(overlay, width=3.35 * inch, height=2.7 * inch))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("КЛАССЫ ТКАНЕЙ", styles["label"]))

    shown = data.class_areas[:_MAX_CLASSES_SHOWN]
    rest = data.class_areas[_MAX_CLASSES_SHOWN:]

    rows = [["Класс", "Доля площади"]]
    for c in shown:
        label = Paragraph(f'<font color="{c.color}">●</font>&nbsp;&nbsp;{c.name_ru}', styles["table_cell"])
        rows.append([label, f"{c.fraction:.1%}"])
    classes_table = Table(rows, colWidths=[2.2 * inch, 1.15 * inch])
    classes_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
        ("FONTNAME", (1, 1), (1, -1), _FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), _MUTED_TEXT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, _BORDER),
        ("LINEBELOW", (0, 1), (-1, -2), 0.5, _BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(classes_table)

    if rest:
        rest_fraction = sum(c.fraction for c in rest)
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(
            f"прочие классы (ещё {len(rest)}): суммарно {rest_fraction:.1%}", styles["small"]
        ))
    return elements


def _build_verdict_panel(data: ReportData, styles: dict) -> Table:
    verdict_color = _MALIGNANT_COLOR if data.is_malignant else _BENIGN_COLOR
    verdict_style = ParagraphStyle("verdict_colored", parent=styles["verdict"], textColor=verdict_color)

    content = [
        Paragraph("ВЕРДИКТ МОДЕЛИ", styles["label"]),
        Paragraph(data.verdict_label, verdict_style),
        Paragraph(f"доля опухолевой ткани <b>{data.tumor_area_fraction:.1%}</b>", styles["verdict_sub"]),
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
