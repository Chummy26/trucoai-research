"""Receipts of the actual policy input, using only legitimate observations."""
import dataclasses
import hashlib
import json


def decision_receipt(obs, reader, match_no):
    from ..agents.learned import FEATURE_VERSION, features
    observation = dataclasses.asdict(obs)
    canonical = json.dumps(observation, sort_keys=True, separators=(',', ':')).encode()
    encoded = features(obs)
    return {'match_no': match_no, 'hand_no': obs.hand_no,
            'observation': observation,
            'observation_sha256': hashlib.sha256(canonical).hexdigest(),
            'features_version': FEATURE_VERSION,
            'features_shape': list(encoded.shape),
            'features_sha256': hashlib.sha256(encoded.tobytes()).hexdigest(),
            'public_context': reader.public_context()}
