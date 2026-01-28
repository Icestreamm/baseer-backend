"""
PDF Generator - Ported from Python V3.8 code
Generates invoice and analysis PDFs matching the original style
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from app.config import config


# Dark blue color matching Python code
DARK_BLUE = colors.HexColor('#1b4a6b')
LIGHT_BLUE_GRAY = colors.HexColor('#e8f4f8')
MEDIUM_GRAY = colors.HexColor('#CCCCCC')


def generate_invoice_pdf(
    report_id: str,
    pdf_path: str,
    photo_results: List[Dict],
    cost_data: Dict,
    report_data: Dict,
    logo_path: str = None
) -> str:
    """
    Generate invoice PDF matching Python V3.8 style.
    
    Args:
        report_id: Assessment ID or report ID
        pdf_path: Full path where PDF should be saved
        photo_results: List of photo processing results with consensus_path
        cost_data: Cost calculation data
        report_data: Additional report metadata
        
    Returns:
        Path to generated PDF
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=1.5*inch,
            bottomMargin=1*inch
        )
        styles = getSampleStyleSheet()
        
        # Custom styles (matching Python code)
        invoice_title_style = ParagraphStyle(
            name='InvoiceTitle',
            parent=styles['Title'],
            fontSize=36,
            textColor=colors.whitesmoke,
            alignment=TA_LEFT,
            spaceAfter=0,
            leading=40
        )
        
        footer_text_style = ParagraphStyle(
            name='FooterText',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.whitesmoke,
            alignment=TA_CENTER,
            spaceAfter=0,
            leading=12
        )
        
        small_caption_style = ParagraphStyle(
            name='Small',
            parent=styles['Normal'],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER
        )
        
        story = []
        
        # Header and Footer functions
        def _header_footer(canvas, doc):
            canvas.saveState()
            # Header
            canvas.setFillColor(DARK_BLUE)
            canvas.rect(0, A4[1] - 1.25*inch, A4[0], 1.25*inch, fill=1)
            
            # "INVOICE" text
            invoice_para = Paragraph("<b>INVOICE</b>", invoice_title_style)
            invoice_para.wrapOn(canvas, 3*inch, 0.5*inch)
            invoice_para.drawOn(canvas, 0.75*inch, A4[1] - 1*inch)
            
            # Logo (if provided)
            logo_width = 1.5 * inch
            logo_height = 1.0 * inch
            
            if logo_path and os.path.exists(logo_path):
                try:
                    logo = RLImage(logo_path, width=logo_width, height=logo_height)
                    logo_x = A4[0] - logo_width - 0.75*inch
                    logo_y = A4[1] - 0.75*inch - logo_height
                    logo.drawOn(canvas, logo_x, logo_y)
                except Exception as e:
                    print(f"Warning: Could not add logo: {e}")
            
            # Footer
            canvas.setFillColor(DARK_BLUE)
            canvas.rect(0, 0, A4[0], 0.75*inch, fill=1)
            footer_para = Paragraph("Thank you for your business!", footer_text_style)
            footer_para.wrapOn(canvas, A4[0], 0.75*inch)
            footer_para.drawOn(canvas, (A4[0] - footer_para.width) / 2, 0.25*inch)
            
            canvas.restoreState()
        
        story.append(Spacer(1, 0.75*inch))
        
        # Invoice Details and Bill To
        invoice_date = report_data.get('date', datetime.now().strftime("%Y-%m-%d"))
        
        invoice_details_left_data = [
            [Paragraph("<b>Invoice No.</b>", styles['Normal']), Paragraph(report_id, styles['Normal'])],
            [Paragraph("<b>Date of Issue</b>", styles['Normal']), Paragraph(invoice_date, styles['Normal'])],
            [Paragraph("<b>Due Date</b>", styles['Normal']), Paragraph("Enter Due Date Here", styles['Normal'])],
        ]
        
        invoice_details_left_table = Table(invoice_details_left_data, colWidths=[1.5*inch, 2.5*inch])
        invoice_details_left_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('FONTSIZE', (0,0), (-1,-1), 10),
        ]))
        
        customer_name = report_data.get('customer_name', 'Client Company Name')
        country = report_data.get('country', 'Address')
        
        bill_to_right_data = [
            [Paragraph("<b>Bill To</b>", styles['h3'])],
            [Paragraph(customer_name, styles['Normal'])],
            [Paragraph(country, styles['Normal'])],
            [Paragraph("Phone", styles['Normal'])],
            [Paragraph("Email", styles['Normal'])],
        ]
        
        bill_to_right_table = Table(bill_to_right_data, colWidths=[3*inch])
        bill_to_right_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('FONTSIZE', (0,0), (-1,-1), 10),
        ]))
        
        main_info_table = Table([[invoice_details_left_table, bill_to_right_table]], colWidths=[4*inch, 3*inch])
        main_info_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0.5*inch),
            ('RIGHTPADDING', (0,0), (-1,-1), 0.5*inch),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(main_info_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Separator line
        story.append(Table([['']], colWidths=[A4[0]-1*inch], style=TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1, colors.HexColor('#AAAAAA'))
        ])))
        story.append(Spacer(1, 0.2*inch))
        
        # Cost Breakdown Table
        currency_symbol = cost_data.get('currency', 'JOD')
        cost_breakdown_data = [
            [Paragraph("<b>Item</b>", styles['Normal']), 
             Paragraph("<b>Description</b>", styles['Normal']), 
             Paragraph("<b>Amount</b>", styles['Normal'])],
        ]
        
        item_count = 1
        # Paint costs per photo
        paint_costs = cost_data.get('paint_costs_local', [])
        if isinstance(paint_costs, list):
            for item in paint_costs:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    photo_num, area, cost = item[0], item[1], item[2]
                elif isinstance(item, dict):
                    photo_num = item.get('photo_num', item_count)
                    cost = item.get('cost', 0)
                else:
                    continue
                    
                if cost > 0:
                    cost_breakdown_data.append([
                        Paragraph(str(item_count), styles['Normal']),
                        Paragraph(f"Damage Repair Photo {photo_num}", styles['Normal']),
                        Paragraph(f"{cost:.2f}", styles['Normal'])
                    ])
                    item_count += 1
        
        # Other costs
        if cost_data.get('light_cost_local', 0) > 0:
            cost_breakdown_data.append([
                Paragraph(str(item_count), styles['Normal']),
                Paragraph("Lights Repair", styles['Normal']),
                Paragraph(f"{cost_data['light_cost_local']:.2f}", styles['Normal'])
            ])
            item_count += 1
            
        if cost_data.get('windshield_cost_local', 0) > 0:
            cost_breakdown_data.append([
                Paragraph(str(item_count), styles['Normal']),
                Paragraph("Windshield Replacement", styles['Normal']),
                Paragraph(f"{cost_data['windshield_cost_local']:.2f}", styles['Normal'])
            ])
            item_count += 1
            
        if cost_data.get('tire_cost_local', 0) > 0:
            cost_breakdown_data.append([
                Paragraph(str(item_count), styles['Normal']),
                Paragraph("Tire Replacement", styles['Normal']),
                Paragraph(f"{cost_data['tire_cost_local']:.2f}", styles['Normal'])
            ])
            item_count += 1
        
        # Add empty rows if needed (to match style)
        for _ in range(max(0, 4 - (len(cost_breakdown_data) - 1))):
            cost_breakdown_data.append([
                Paragraph("", styles['Normal']),
                Paragraph("", styles['Normal']),
                Paragraph("", styles['Normal'])
            ])
        
        cost_breakdown_table = Table(cost_breakdown_data, colWidths=[0.5*inch, 3.5*inch, 2*inch])
        cost_breakdown_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), MEDIUM_GRAY),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('ALIGN', (2,0), (2,-1), 'RIGHT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#AAAAAA')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BLUE_GRAY]),
        ]))
        story.append(cost_breakdown_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Summary section
        subtotal_local_base = cost_data.get('subtotal_local_base', 0)
        tax_rate = cost_data.get('tax_rate', 0)
        tax_amount_local = cost_data.get('tax_amount_on_base_local', 0)
        final_local_cost = cost_data.get('final_local_cost', 0)
        
        summary_data = [
            [Paragraph("Subtotal", styles['Normal']), Paragraph(f"{subtotal_local_base:.2f}", styles['Normal'])],
            [Paragraph("Discount", styles['Normal']), Paragraph("$0.00", styles['Normal'])],
            [Paragraph("Tax Rate", styles['Normal']), Paragraph(f"{tax_rate*100:.2f}%", styles['Normal'])],
            [Paragraph("Tax", styles['Normal']), Paragraph(f"{tax_amount_local:.2f}", styles['Normal'])],
            [Paragraph("<b>Total</b>", styles['Normal']), Paragraph(f"<b>{final_local_cost:.2f}</b>", styles['Normal'])],
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1*inch])
        summary_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'RIGHT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
            ('LINEABOVE', (0,-1), (-1,-1), 1, colors.HexColor('#AAAAAA')),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('BACKGROUND', (1,-1), (1,-1), LIGHT_BLUE_GRAY),
        ]))
        
        right_aligned_summary = Table([['', summary_table]], colWidths=[4*inch, 3*inch])
        right_aligned_summary.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'RIGHT'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(right_aligned_summary)
        story.append(Spacer(1, 0.5*inch))
        
        # Separator line
        story.append(Table([['']], colWidths=[A4[0]-1*inch], style=TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1, colors.HexColor('#AAAAAA'))
        ])))
        story.append(Spacer(1, 0.2*inch))
        
        # Photos section
        story.append(Paragraph("<b>Damage Assessment Photos</b>", styles['h3']))
        story.append(Paragraph("The following photos show detected damage areas marked with bounding boxes:", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        elements_for_photos = []
        photos_in_row = []
        
        for photo_result in photo_results:
            photo_num = photo_result.get('photo_num', 1)
            consensus_path = photo_result.get('consensus_path')
            original_path = photo_result.get('photo_path')
            
            image_to_display = consensus_path if consensus_path and os.path.exists(consensus_path) else original_path
            image_title = f"Photo {photo_num}: Damage Detection" if consensus_path else f"Photo {photo_num}: Original"
            
            if image_to_display and os.path.exists(image_to_display):
                try:
                    img = RLImage(image_to_display, width=3.2*inch, height=2.4*inch)
                    img_caption = Paragraph(image_title, small_caption_style)
                    photos_in_row.append([img, img_caption])
                except Exception as e:
                    print(f"Warning: Could not add image {image_to_display}: {e}")
                    photos_in_row.append([
                        Paragraph("[Image not found]", small_caption_style),
                        Paragraph(image_title + " (file not found)", small_caption_style)
                    ])
            else:
                photos_in_row.append([
                    Paragraph("[Image not found]", small_caption_style),
                    Paragraph(image_title + " (file not found)", small_caption_style)
                ])
            
            if len(photos_in_row) == 2:
                img_block_table = Table(
                    [[p[0] for p in photos_in_row], [p[1] for p in photos_in_row]],
                    colWidths=[3.5*inch, 3.5*inch],
                    rowHeights=[2.4*inch, None]
                )
                img_block_table.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('LEFTPADDING', (0,0), (-1,-1), 5),
                    ('RIGHTPADDING', (0,0), (-1,-1), 5),
                    ('TOPPADDING', (0,0), (-1,-1), 5),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ]))
                elements_for_photos.append(img_block_table)
                elements_for_photos.append(Spacer(1, 0.2*inch))
                photos_in_row = []
        
        # Add remaining photos if odd number
        if photos_in_row:
            if len(photos_in_row) == 1:
                photos_in_row.append([Spacer(1, 0.1), Spacer(1, 0.1)])
            img_block_table = Table(
                [[p[0] for p in photos_in_row], [p[1] for p in photos_in_row]],
                colWidths=[3.5*inch, 3.5*inch],
                rowHeights=[2.4*inch, None]
            )
            img_block_table.setStyle(TableStyle([
                ('BOX', (0,0), (-1,-1), 1, colors.black),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('LEFTPADDING', (0,0), (-1,-1), 5),
                ('RIGHTPADDING', (0,0), (-1,-1), 5),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ]))
            elements_for_photos.append(img_block_table)
            elements_for_photos.append(Spacer(1, 0.2*inch))
        
        story.extend(elements_for_photos)
        story.append(Spacer(1, 0.5*inch))
        
        # Build PDF
        doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
        
        print(f"✅ Invoice PDF generated: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        print(f"❌ Error generating invoice PDF: {e}")
        import traceback
        traceback.print_exc()
        raise


def generate_analysis_pdf(
    report_id: str,
    pdf_path: str,
    analysis_text: str,
    photo_results: List[Dict]
) -> str:
    """
    Generate analysis PDF with detailed text and all photos.
    
    Args:
        report_id: Assessment ID
        pdf_path: Full path where PDF should be saved
        analysis_text: Text analysis output
        photo_results: List of photo results with all annotation paths
        
    Returns:
        Path to generated PDF
    """
    try:
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            topMargin=1*inch,
            bottomMargin=1*inch
        )
        styles = getSampleStyleSheet()
        
        mono_style = ParagraphStyle(
            name='Mono',
            fontName='Courier',
            fontSize=8,
            leading=9,
            alignment=TA_LEFT,
            wordWrap='CJK'
        )
        
        story = []
        
        def _create_framed_image_block(image_path, title_text, width=3.5*inch, height=2.625*inch):
            elements = []
            if image_path and os.path.exists(image_path):
                try:
                    img = RLImage(image_path, width=width, height=height)
                    elements.append(img)
                except Exception as e:
                    print(f"Warning: Could not add image {image_path}: {e}")
                    elements.append(Paragraph("[Image not found]", styles['Normal']))
            else:
                elements.append(Paragraph("[Image not found]", styles['Normal']))
            
            elements.append(Paragraph(title_text, styles['Normal']))
            
            img_table = Table([[elem] for elem in elements])
            img_table.setStyle(TableStyle([
                ('BOX', (0,0), (-1,-1), 1, colors.black),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('LEFTPADDING', (0,0), (-1,-1), 5),
                ('RIGHTPADDING', (0,0), (-1,-1), 5),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ]))
            return img_table
        
        # Title
        story.append(Paragraph(f"Analysis Report: {report_id}", styles['Title']))
        story.append(Spacer(1, 0.2*inch))
        
        # Analysis text
        if analysis_text:
            for line in analysis_text.split('\n'):
                if line.strip():  # Skip empty lines
                    story.append(Paragraph(line, mono_style))
                    story.append(Spacer(1, 0.05*inch))
        
        # Photos section
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("All Processed Photos", styles['h2']))
        
        for photo_result in photo_results:
            photo_num = photo_result.get('photo_num', 1)
            original_path = photo_result.get('photo_path')
            
            # Original image
            if original_path:
                story.append(_create_framed_image_block(original_path, f"Photo {photo_num}: Original", styles))
                story.append(Spacer(1, 0.1*inch))
            
            # Consensus image if available
            consensus_path = photo_result.get('consensus_path')
            if consensus_path:
                story.append(_create_framed_image_block(
                    consensus_path,
                    f"Photo {photo_num}: Final Consensus Damage",
                    styles
                ))
                story.append(Spacer(1, 0.1*inch))
            
            story.append(Spacer(1, 0.3*inch))
        
        doc.build(story)
        print(f"✅ Analysis PDF generated: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        print(f"❌ Error generating analysis PDF: {e}")
        import traceback
        traceback.print_exc()
        raise
