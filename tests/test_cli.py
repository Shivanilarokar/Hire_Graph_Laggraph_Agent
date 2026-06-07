"""Canned demo scenario tests."""

from hiregraph.cli import SCENARIOS


def test_shivani_pdf_is_in_canned_scenarios():
    resume_names = [resume.name for _label, resume, _jd in SCENARIOS]

    assert "Shivani_Resume.pdf" in resume_names
