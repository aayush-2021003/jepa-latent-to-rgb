"""Run V-JEPA 2.1 four-frame prediction and Cosmos decoding."""
from experiments.jepa_cosmos import infer as implementation
from experiments.vjepa21_cosmos.common import load_config
from experiments.vjepa21_cosmos.models import (
    OfficialVJEPA21WorldModel,
    validate_model_geometry,
)


def main() -> None:
    implementation.load_config = load_config
    implementation.validate_model_geometry = validate_model_geometry
    implementation.FactorJEPAWorldModel = OfficialVJEPA21WorldModel
    implementation.main()


if __name__ == "__main__":
    main()
