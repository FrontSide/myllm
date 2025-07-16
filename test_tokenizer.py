from tokenizer import Tokenizer

def test_encoding():
    t = Tokenizer()
    assert t.encode("Hello there.") == [15496, 612, 13]

def test_decoding():
    t = Tokenizer()
    assert t.decode([50, 60]) == "S]"


