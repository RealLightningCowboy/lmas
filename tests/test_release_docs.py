from lmas.help_docs import DEVELOPMENT_PROVENANCE, POLARITY_PRODUCT_FORMAT, read_help_document


def test_development_provenance_is_packaged():
    text = read_help_document(DEVELOPMENT_PROVENANCE)
    assert 'R. Stetson Reger' in text
    assert 'ChatGPT Plus' in text


def test_polarity_product_format_is_packaged():
    text = read_help_document(POLARITY_PRODUCT_FORMAT)
    assert "lmas-polarity-v1" in text
    assert "dataset fingerprint" in text.lower()
