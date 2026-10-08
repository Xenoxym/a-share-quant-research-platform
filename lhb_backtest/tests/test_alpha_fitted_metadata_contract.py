"""The producer's pair hash format must reach the account without changing receipts."""
import pytest
from src.alpharesearch.audit import _model_code_hashes

@pytest.mark.parametrize("shape",["dictionary","tuple","json_list"])
def test_model_hashes_accept_existing_producer_and_legacy_stub_formats(shape):
    key="src/alpharesearch/models.py";sha="a"*64
    value={"dictionary":{key:sha},"tuple":((key,sha),),"json_list":[[key,sha]]}[shape]
    assert _model_code_hashes(value)=={key:sha}
    assert type(value) is {"dictionary":dict,"tuple":tuple,"json_list":list}[shape]

@pytest.mark.parametrize("value",[[["src/alpharesearch/models.py","a"*64]]*2,
    [["src/alpharesearch/models.py",True]],[["other.py","a"*64]],[["src/alpharesearch/models.py","z"*64]]])
def test_model_hashes_reject_duplicate_bad_type_path_or_digest(value):
    with pytest.raises(ValueError):_model_code_hashes(value)
