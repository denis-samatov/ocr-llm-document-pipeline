import numpy as np
import pytest
import cv2

from ocr_llm_pipeline import get_ollama_api_key, iter_input_files, preprocess_image


def test_iter_input_files_filters_by_extension_and_sorts(tmp_path):
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "c.jpg").write_bytes(b"")
    (tmp_path / "ignored.txt").write_bytes(b"")
    (tmp_path / "subdir").mkdir()

    files = list(iter_input_files(tmp_path))

    assert [f.name for f in files] == ["a.pdf", "b.png", "c.jpg"]


def test_iter_input_files_on_empty_directory(tmp_path):
    assert list(iter_input_files(tmp_path)) == []


def test_preprocess_image_returns_a_binarized_temp_file(tmp_path):
    image = np.full((50, 50, 3), 200, dtype=np.uint8)
    image_path = tmp_path / "input.png"
    cv2.imwrite(str(image_path), image)

    output_path = preprocess_image(image_path)

    result = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    assert result is not None
    # adaptiveThreshold output is binary: only 0 and 255 values
    assert set(np.unique(result)).issubset({0, 255})


def test_preprocess_image_raises_on_unreadable_file(tmp_path):
    bad_path = tmp_path / "not_an_image.png"
    bad_path.write_bytes(b"not a real image")

    with pytest.raises(ValueError, match="Could not read image"):
        preprocess_image(bad_path)


def test_get_ollama_api_key_reads_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key-123")

    assert get_ollama_api_key() == "test-key-123"


def test_get_ollama_api_key_raises_when_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Ollama API key is not configured"):
        get_ollama_api_key()
