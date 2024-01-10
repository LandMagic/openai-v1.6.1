import os
import pytest
from data_utils import get_embedding

# Envrioment variables have to be passed in for the test
# Endpoint gets truncated during import, and api version needs to be added manually
endpoint = os.environ.get('EMBEDDING_MODEL_ENDPOINT') + '=2023-08-01-preview'
key = os.environ.get('EMBEDDING_MODEL_KEY')

embeddings = get_embedding('Reseearch is important for humanity.',embedding_model_endpoint=endpoint, embedding_model_key=key)

def test_get_embeddings_is_not_None():
    print(endpoint)
    assert embeddings is not None

def test_get_embeddings_is_not_empty():
    assert not len(embeddings) == 0

def test_get_embeddings_is_list():
    assert isinstance(embeddings, list)

def test_get_embeddings_is_between_neg1_and_1():
    assert (-1 <= embeddings[0] <= 1)

def test_empty_text_prompt():
    with pytest.raises(Exception) as e:
        get_embedding('', embedding_model_endpoint=endpoint, embedding_model_key=key)
    assert str(e.value) is not None