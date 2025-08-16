from cmor import severity_label

def test_severity_label():
    assert severity_label(2.9) == "ниже порога"
    assert severity_label(3.0) == "лёгкая"
    assert severity_label(15.0) == "умеренная"
    assert severity_label(25.0) == "тяжёлая"
