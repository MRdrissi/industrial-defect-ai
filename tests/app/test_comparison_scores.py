"""La comparaison ne confond pas les scores bruts — US-06-05 (IDL-80).

Scénarios où comparer les scores entre eux donnerait une conclusion fausse :
chaque décision doit dépendre uniquement du score de la méthode et de SON seuil,
et l'écran ne doit jamais associer le score d'une méthode au seuil de l'autre.

Les valeurs de seuil reprennent l'ordre de grandeur des seuils calibrés en
US-05-02 pour Transistor (CAE 0,0274 ; PatchCore 40,96).
"""

import pytest
import torch
from streamlit.testing.v1 import AppTest

from app.comparison import decisions_summary, run_comparison
from src.inference.inference_contract import InferenceResult

CAE_THRESHOLD = 0.027364758774638176
PATCHCORE_THRESHOLD = 40.95855598449707


def result(method, score, threshold):
    return InferenceResult(
        category="transistor",
        method=method,
        version="exp-001",
        score=score,
        threshold=threshold,
        anomaly_map=torch.zeros(16, 16),
        decision="suspect" if score >= threshold else "normal",
        latency_ms=1.0,
        simulated=False,
    )


def fake_registry(cae_threshold, patchcore_threshold):
    """Registre simulé : même modèle factice, mais un seuil propre à chaque méthode."""
    entry = {"category": "transistor", "version": "demo", "backend": "fake", "artifact_path": None}
    return {
        ("transistor", "cae", "demo"): {**entry, "method": "cae", "threshold": cae_threshold},
        ("transistor", "patchcore", "demo"): {**entry, "method": "patchcore", "threshold": patchcore_threshold},
    }


def test_each_decision_uses_its_own_threshold():
    # Même image, même score factice (0,2) pour les deux méthodes ;
    # seuils différents → décisions différentes.
    image = torch.full((3, 8, 8), 0.7)  # |0,7 - 0,5| = 0,2
    comparison = run_comparison(
        image, "transistor", {"cae": "demo", "patchcore": "demo"}, fake_registry(0.1, 0.3)
    )

    assert comparison["cae"].score == pytest.approx(comparison["patchcore"].score)
    assert comparison["cae"].decision == "suspect"       # 0,2 >= 0,1
    assert comparison["patchcore"].decision == "normal"  # 0,2 <  0,3
    assert "CAE → SUSPECT, PatchCore → NORMAL" in decisions_summary(comparison)


@pytest.mark.parametrize(
    "cae_score, patchcore_score, expected",
    [
        # PatchCore a le score « le plus grand » mais il est sous SON seuil : normal.
        (0.0300, 39.0, "CAE → SUSPECT, PatchCore → NORMAL"),
        # CAE a le score « le plus petit » mais il dépasse SON seuil : suspect.
        (0.0280, 41.5, "Les deux méthodes concluent : SUSPECT."),
        (0.0100, 20.0, "Les deux méthodes concluent : NORMAL."),
    ],
)
def test_magnitude_of_raw_scores_never_drives_the_decision(cae_score, patchcore_score, expected):
    results = {
        "cae": result("cae", cae_score, CAE_THRESHOLD),
        "patchcore": result("patchcore", patchcore_score, PATCHCORE_THRESHOLD),
    }

    def infer(request, _registry):
        return results[request.method]

    comparison = run_comparison(
        torch.rand(3, 8, 8), "transistor", {"cae": "exp-001", "patchcore": "exp-001"}, {}, infer
    )
    assert expected in decisions_summary(comparison)


def _realistic_disagreement_screen():
    """Script autonome : CAE au-dessus de son seuil, PatchCore sous le sien."""
    import torch
    from PIL import Image

    from app.comparison import ComparisonResult
    from app.comparison_view import display_comparison
    from src.inference.inference_contract import InferenceResult

    def make(method, score, threshold, decision):
        return InferenceResult(
            category="transistor", method=method, version="exp-001", score=score,
            threshold=threshold, anomaly_map=torch.rand(256, 256), decision=decision,
            latency_ms=5.0, simulated=False,
        )

    comparison = ComparisonResult(
        category="transistor",
        image_shape=(3, 1024, 1024),
        results={
            "cae": make("cae", 0.0300, 0.027364758774638176, "suspect"),
            "patchcore": make("patchcore", 39.0, 40.95855598449707, "normal"),
        },
    )
    display_comparison(comparison, Image.new("RGB", (1024, 1024)))


def test_screen_pairs_each_score_with_its_own_threshold():
    at = AppTest.from_function(_realistic_disagreement_screen).run()
    captions = [element.value for element in at.caption]
    cae_line = next(text for text in captions if text.startswith("Score brut : 0.0300"))
    patchcore_line = next(text for text in captions if text.startswith("Score brut : 39.0000"))

    assert not at.exception
    assert "seuil CAE : 0.0274" in cae_line and "PatchCore" not in cae_line
    assert "seuil PatchCore : 40.9586" in patchcore_line and "CAE" not in patchcore_line

    assert len(at.warning) == 1 and "SUSPECT" in at.warning[0].value   # CAE
    assert len(at.success) == 1 and "NORMAL" in at.success[0].value    # PatchCore
    assert "CAE → SUSPECT, PatchCore → NORMAL" in " ".join(m.value for m in at.markdown)
    assert any("pas comparables" in info.value for info in at.info)
