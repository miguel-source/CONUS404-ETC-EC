"""We assemble vector supplemental figures and captions into a review PDF.

Run with the reportlab/pypdf/Pillow environment. We render the assembled PDF
with pypdfium2 for visual quality assurance when Poppler is unavailable.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas


PAGE_WIDTH, PAGE_HEIGHT = 17 * 72, 11 * 72
BROWN = HexColor('#A6611A')
GREEN = HexColor('#01665E')


def font_setup():
    windows = Path('C:/Windows/Fonts')
    regular, bold = windows / 'times.ttf', windows / 'timesbd.ttf'
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont('StudyTimes', str(regular)))
        pdfmetrics.registerFont(TTFont('StudyTimesBold', str(bold)))
        return 'StudyTimes', 'StudyTimesBold'
    return 'Times-Roman', 'Times-Bold'


def clean_caption(value: str) -> str:
    value = re.sub(r'^#+\s*', '', value.strip(), flags=re.M)
    value = re.sub(r'\*\*([^*]+)\*\*', r'\1', value)
    value = re.sub(r'`([^`]+)`', r'\1', value)
    value = value.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-')
    return value


def paragraph(canvas, text, x, top, width, font, size=12, leading=17):
    style = ParagraphStyle('caption', fontName=font, fontSize=size, leading=leading)
    p = Paragraph(escape(text).replace('\n', '<br/>'), style)
    _, height = p.wrap(width, PAGE_HEIGHT)
    p.drawOn(canvas, x, top - height)
    return height


def page_base(page_number, section, font, bold):
    memory = io.BytesIO()
    canvas = Canvas(memory, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    canvas.setFillColor(GREEN)
    canvas.setFont(bold, 13)
    canvas.drawString(42, PAGE_HEIGHT - 32, 'Tropical cyclone precipitation evaluation')
    canvas.setFillColor(HexColor('#444444'))
    canvas.setFont(font, 10)
    canvas.drawRightString(PAGE_WIDTH - 42, PAGE_HEIGHT - 32, section)
    canvas.setStrokeColor(HexColor('#C6C6C6'))
    canvas.setLineWidth(0.5)
    canvas.line(42, PAGE_HEIGHT - 42, PAGE_WIDTH - 42, PAGE_HEIGHT - 42)
    canvas.setFont(font, 10)
    canvas.drawString(42, 23, 'Supplementary material | We report conditional estimates on the stated spatial subsets.')
    canvas.drawRightString(PAGE_WIDTH - 42, 23, str(page_number))
    return memory, canvas


def assemble(root: Path, manifest_path: Path, output: Path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    figures = manifest.get('figures', [])
    if isinstance(figures, dict):
        figures = [dict(value, id=key) for key, value in figures.items()]
    if not figures:
        raise ValueError('Manifest does not contain figures')
    font, bold = font_setup()
    writer = PdfWriter()
    writer.add_metadata({'/Title': 'Supplementary figures: tropical cyclone precipitation collocation',
                         '/Author': 'Miguel Diaz', '/Creator': 'tc-precip-collocation'})

    memory, canvas = page_base(1, 'Supplementary figure index', font, bold)
    canvas.setFillColor(BROWN)
    canvas.setFont(bold, 25)
    canvas.drawString(60, PAGE_HEIGHT - 95, 'Supplementary figures and computational diagnostics')
    intro = (
        'We provide additional diagnostics for our comparison of CONUS404, CONUS404BA, '
        'MRMS Radar-Only QPE, and PERSIANN-CDR across ten tropical cyclones and 23 watersheds. '
        'The figures use the study\'s brown-to-green palette and Times New Roman typography. '
        'Individual 600-dpi PNG/TIFF and vector PDF/SVG files accompany this review copy.'
    )
    canvas.setFillColor(HexColor('#111111'))
    paragraph(canvas, intro, 60, PAGE_HEIGHT - 125, PAGE_WIDTH - 120, font, 14, 21)
    cursor = PAGE_HEIGHT - 220
    for figure in figures:
        fid = figure.get('id', figure.get('figure', ''))
        title = figure.get('title', str(fid).replace('_', ' '))
        canvas.setFont(bold, 15)
        canvas.setFillColor(GREEN)
        canvas.drawString(65, cursor, f'{fid}. {title}')
        cursor -= 38
    paragraph(canvas,
        'We distinguish sample eligibility, point admissibility, bootstrap retention, and common-triplet '
        'admissibility. Raw error SD has units of mm per day; log1p error SD has transformed units. '
        'Only the CONUS404-CONUS404BA error correlation is estimated by the four-product EC model. '
        'Gray map cells have the meaning specified in each caption; they do not represent zero error.',
        60, cursor - 12, PAGE_WIDTH - 120, font, 13, 19)
    paragraph(canvas,
        'Map context: Natural Earth and USGS Watershed Boundary Dataset derivatives. '
        'We document the source files, spatial subsets, and computational provenance in the release package.',
        60, 100, PAGE_WIDTH - 120, font, 11, 15)
    canvas.showPage()
    canvas.save()
    memory.seek(0)
    writer.add_page(PdfReader(memory).pages[0])

    checks = []
    for number, figure in enumerate(figures, start=2):
        fid = str(figure.get('id', figure.get('figure', f'S{number-1}')))
        files = figure.get('files', {})
        pdf_path = root / figure.get('pdf', files.get('pdf', ''))
        caption = figure.get('caption', '')
        if figure.get('caption_md'):
            caption = (root / figure['caption_md']).read_text(encoding='utf-8')
        elif isinstance(caption, str) and '\n' not in caption and len(caption) < 250 and (root / caption).is_file():
            caption = (root / caption).read_text(encoding='utf-8')
        caption = clean_caption(str(caption))
        if not caption:
            raise ValueError(f'Missing caption: {fid}')
        memory, canvas = page_base(number, f'Figure {fid}', font, bold)
        caption_height = paragraph(canvas, caption, 48, 150, PAGE_WIDTH - 96, font, 11.5, 15.5)
        if caption_height > 110:
            raise ValueError(f'Caption too tall for {fid}: {caption_height}')
        canvas.showPage()
        canvas.save()
        memory.seek(0)
        page = PdfReader(memory).pages[0]
        source = PdfReader(str(pdf_path)).pages[0]
        width, height = float(source.mediabox.width), float(source.mediabox.height)
        available_width = PAGE_WIDTH - 78
        bottom, top = 165, PAGE_HEIGHT - 57
        scale = min(available_width / width, (top - bottom) / height)
        tx = (PAGE_WIDTH - width * scale) / 2
        ty = bottom + ((top - bottom) - height * scale) / 2
        page.merge_transformed_page(source, Transformation().scale(scale).translate(tx, ty))
        writer.add_page(page)
        checks.append({'figure': fid, 'vector_source': str(pdf_path.relative_to(root)),
                       'scale': scale, 'caption_height_pt': caption_height,
                       'source_width_pt': width, 'source_height_pt': height})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        writer.write(stream)
    check_reader = PdfReader(str(output))
    if len(check_reader.pages) != len(figures) + 1:
        raise AssertionError('Unexpected booklet page count')
    extracted = '\n'.join(page.extract_text() or '' for page in check_reader.pages)
    if '\ufffd' in extracted:
        raise AssertionError('Replacement glyph in assembled text')
    (output.parent / 'appendix_pdf_checks.json').write_text(json.dumps({
        'pages': len(check_reader.pages), 'page_size_inches': [17, 11],
        'figure_count': len(figures), 'vector_figures_preserved': True,
        'font': font, 'checks': checks,
    }, indent=2), encoding='utf-8')
    return output


def render_review(pdf_path: Path):
    import pypdfium2 as pdfium
    destination = pdf_path.parent / 'review'
    destination.mkdir(exist_ok=True)
    document = pdfium.PdfDocument(str(pdf_path))
    thumbnails = []
    for index in range(len(document)):
        page = document[index]
        image = page.render(scale=1.25).to_pil().convert('RGB')
        image.save(destination / f'page_{index+1:02d}.png')
        image.thumbnail((850, 550))
        thumbnails.append(image.copy())
        page.close()
    document.close()
    columns, tile_w, tile_h = 2, 880, 590
    rows = (len(thumbnails) + 1) // 2
    sheet = Image.new('RGB', (columns * tile_w, rows * tile_h), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, thumbnail in enumerate(thumbnails):
        x, y = (index % columns) * tile_w, (index // columns) * tile_h
        draw.text((x + 12, y + 8), f'Page {index+1}', fill='black')
        sheet.paste(thumbnail, (x + 12, y + 30))
    sheet.save(destination / 'appendix_contact_sheet.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'results' / 'supplementary')
    parser.add_argument('--manifest', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results' / 'supplementary_figures_review.pdf')
    args = parser.parse_args()
    manifest_path = args.manifest or args.root / 'manifest.json'
    result = assemble(args.root, manifest_path, args.output)
    render_review(result)
    print(result)
