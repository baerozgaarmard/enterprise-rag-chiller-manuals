"""Fixtures and synthetic chiller manuals for RAG testing."""

from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def synthetic_chiller_manual_pdf(tmp_path: Path) -> Path:
    """Generate a 2-page synthetic technical chiller manual."""
    pdf_path = tmp_path / "model_x500_manual.pdf"
    doc = pymupdf.open()

    # Page 1: General Specs
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        pymupdf.Point(50, 70),
        "Centrifugal Water Chiller Model X500 - Technical Specifications",
        fontsize=16,
    )
    p1.insert_text(
        pymupdf.Point(50, 110),
        "The Model X500 Centrifugal Chiller operates exclusively on R-134a refrigerant with a factory charge of 450 lbs.",
        fontsize=11,
    )
    p1.insert_text(
        pymupdf.Point(50, 140),
        "The nominal cooling capacity is 500 tons at AHRI standard rating conditions.",
        fontsize=11,
    )
    p1.insert_text(
        pymupdf.Point(50, 170),
        "The entering condenser water temperature design basis is 85°F (29.4°C) with a nominal flow rate of 3.0 GPM per ton.",
        fontsize=11,
    )

    # Page 2: Maintenance & Safety Limits
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text(
        pymupdf.Point(50, 70),
        "Safety Guidelines and Operating Pressure Limits",
        fontsize=16,
    )
    p2.insert_text(
        pymupdf.Point(50, 110),
        "Danger: High voltage 460V electrical hazard. Main disconnect switch must be locked out before opening the starter enclosure.",
        fontsize=11,
    )
    p2.insert_text(
        pymupdf.Point(50, 140),
        "Compressor oil sump pressure must remain between 15 and 25 psig during steady compressor operation.",
        fontsize=11,
    )
    p2.insert_text(
        pymupdf.Point(50, 170),
        "High pressure safety cutout trip point is calibrated to 180 psig.",
        fontsize=11,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def synthetic_absorption_manual_pdf(tmp_path: Path) -> Path:
    """Generate a second distinct manual for multi-document isolation testing."""
    pdf_path = tmp_path / "absorption_chiller_a200.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        pymupdf.Point(50, 70),
        "Absorption Chiller Model A200 Engineering Manual",
        fontsize=16,
    )
    p1.insert_text(
        pymupdf.Point(50, 110),
        "The Model A200 Absorption Chiller uses Lithium Bromide (LiBr) solution with deionized water as the working refrigerant.",
        fontsize=11,
    )
    p1.insert_text(
        pymupdf.Point(50, 140),
        "Steam operating supply pressure must not exceed 115 psig.",
        fontsize=11,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path
